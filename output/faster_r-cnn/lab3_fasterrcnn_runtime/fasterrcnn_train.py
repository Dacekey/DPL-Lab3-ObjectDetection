"""Torchvision Faster R-CNN training via torchrun; evaluation runs in a separate process."""
import argparse
import csv
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

from common import (CocoDataset, MAX_DETECTIONS, NMS_IOU, SCORE_FLOOR, Predictor,
                    coco_metrics, detection_collate, final_evaluation, predict_dataset,
                    save_json, seed_everything, seed_worker)


def build_model(pretrained=True, size=640):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None, min_size=size, max_size=size,
        box_score_thresh=SCORE_FLOOR, box_nms_thresh=NMS_IOU,
        box_detections_per_img=MAX_DETECTIONS)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    if not pretrained:
        # Torchvision chooses ordinary BN when both weights arguments are None.
        # Recreate the FrozenBN graph used by the COCO-pretrained training model.
        def freeze_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child,torch.nn.BatchNorm2d):
                    setattr(module,name,FrozenBatchNorm2d(child.num_features,eps=0.0))
                else:
                    freeze_bn(child)
        freeze_bn(model)
    return model


def atomic_save(payload, path):
    temp = Path(str(path) + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def require_finite(values, *, output, rank, epoch, step, targets, optimizer, amp, stage):
    """Stop every rank before an invalid optimizer update; preserve local evidence."""
    finite = torch.stack([torch.isfinite(v.detach()).all() for v in values.values()]).all().int()
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if finite.item():
        return
    def number(value):
        value = float(value.detach())
        return value if math.isfinite(value) else str(value)
    report = {"rank": rank, "epoch": epoch + 1, "step": step, "stage": stage,
              "amp": amp, "values": {k: number(v) for k, v in values.items()},
              "learning_rate": optimizer.param_groups[0]["lr"],
              "targets": [{"image_id": int(t["image_id"]),
                           "boxes": t["boxes"].detach().cpu().tolist(),
                           "labels": t["labels"].detach().cpu().tolist()} for t in targets]}
    path = Path(output) / f"nonfinite_rank{rank}.json"
    save_json(path, report)
    print(json.dumps(report), flush=True)
    raise RuntimeError(f"Non-finite {stage} at epoch={epoch+1}, step={step}; "
                       f"abort all ranks; local diagnostics: {path}")


def train(config):
    try:
        _train(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU T4 x2 in Kaggle settings.")
    torch.cuda.set_device(local)
    device = torch.device("cuda",local)
    if distributed:
        dist.init_process_group("nccl",timeout=timedelta(minutes=60),device_id=device)
    seed_everything(config["seed"]+rank)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(f"Training precision: {'AMP FP16' if config['amp'] else 'FP32'}", flush=True)
    if config["batch"] % world:
        raise ValueError("Global batch must be divisible by GPU count")
    # Rank zero populates the pretrained cache before other ranks read it.
    if distributed and rank != 0:
        dist.barrier()
    model = build_model(True,config["size"]).to(device)
    if distributed and rank == 0:
        dist.barrier()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],momentum=0.9,weight_decay=0.0005)
    train_set = CocoDataset(config["root"],"train",config["size"],augment=True)
    sampler = DistributedSampler(train_set,num_replicas=world,rank=rank,shuffle=True,
                                 seed=config["seed"]) if distributed else None
    generator = torch.Generator().manual_seed(config["seed"]+rank)
    loader = DataLoader(train_set,batch_size=config["batch"]//world,shuffle=sampler is None,
        sampler=sampler,num_workers=config["workers"],pin_memory=True,
        multiprocessing_context="spawn" if config["workers"] else None,
        collate_fn=detection_collate,worker_init_fn=seed_worker,generator=generator)
    steps = len(loader)
    def lr_multiplier(step):
        warmup = max(config["warmup_epochs"]*steps,1)
        if step < warmup:
            return 0.01+0.99*step/warmup
        progress = (step-warmup)/max(config["epochs"]*steps-warmup,1)
        return 0.01+0.99*(1+math.cos(math.pi*min(progress,1)))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
    scaler = torch.amp.GradScaler("cuda",enabled=config["amp"])
    epoch_start, best_ap, bad_epochs = 0,-1.0,0
    history = []
    if config.get("resume"):
        checkpoint = torch.load(config["resume"],map_location="cpu",weights_only=False)
        for key in ("seed","size","batch","epochs"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Resume requires unchanged {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch_start,best_ap,bad_epochs = checkpoint["epoch"]+1,checkpoint["best_ap"],checkpoint["bad_epochs"]
        history = checkpoint.get("history",[])
        if epoch_start >= config["epochs"]:
            raise ValueError("This run already completed; evaluate best.pt instead of resuming.")
        if rank == 0 and not (output/"best.pt").exists():
            raise ValueError("Copy the previous best.pt next to last.pt into the new output before resuming.")
    wrapped = DistributedDataParallel(model,device_ids=[local]) if distributed else model
    val_set = CocoDataset(config["root"],"val",config["size"]) if rank == 0 else None
    start=time.perf_counter()
    for epoch in range(epoch_start,config["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        seed_everything(config["seed"]+epoch*world+rank)
        wrapped.train()
        torch.cuda.reset_peak_memory_stats(device)
        running=torch.zeros(2,device=device)
        components={}
        epoch_start_time=time.perf_counter()
        for step,(images,targets,_) in enumerate(loader):
            images=[image.to(device,non_blocking=True) for image in images]
            targets=[{k:v.to(device,non_blocking=True) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]):
                loss_dict=wrapped(images,targets)
                loss=sum(loss_dict.values())
            diagnostics = dict(output=output, rank=rank, epoch=epoch, step=step,
                               targets=targets, optimizer=optimizer, amp=config["amp"])
            require_finite({**loss_dict, "total_loss": loss}, stage="loss", **diagnostics)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
            # With AMP, GradScaler handles overflowing gradients by skipping the
            # update. FP32 has no such protection, so fail before optimizer.step.
            if not config["amp"]:
                require_finite({**loss_dict, "grad_norm": grad_norm},
                               stage="gradient", **diagnostics)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            running[0]+=loss.detach()
            running[1]+=1
            for k,v in loss_dict.items():
                components[k]=components.get(k,0.0)+float(v.detach())
            if rank == 0 and step % 100 == 0:
                print(f"epoch {epoch+1}/{config['epochs']} step {step}/{steps} loss={loss.item():.4f}",flush=True)
        train_peak=torch.tensor(torch.cuda.max_memory_allocated(device)/1024**2,device=device)
        if distributed:
            dist.all_reduce(running)
            dist.all_reduce(train_peak,op=dist.ReduceOp.MAX)
        control=torch.zeros(3,device=device)
        if rank == 0:
            model.eval()
            predictions=predict_dataset(Predictor(model,"fasterrcnn",device),val_set,
                                        config["eval_batch"],config["workers"])
            metrics=coco_metrics(val_set.annotation_path,predictions)
            ap=metrics["mAP50_95"]
            improved=ap > best_ap
            if improved:
                best_ap,bad_epochs=ap,0
            else:
                bad_epochs+=1
            row={"epoch":epoch+1,"train_loss":float(running[0]/running[1]),
                 "val_mAP50_95":ap,"val_AP50":metrics["AP50"],"val_AP75":metrics["AP75"],
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "train_peak_VRAM_MB":float(train_peak),"epoch_seconds":time.perf_counter()-epoch_start_time,
                 **{k:v/steps for k,v in components.items()}}
            history.append(row)
            payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                     "epoch":epoch,"best_ap":best_ap,"bad_epochs":bad_epochs,
                     "history":history,"config":config}
            atomic_save(payload,output/"last.pt")
            if improved:
                atomic_save(payload,output/"best.pt")
                save_json(output/"best_validation_metrics.json",metrics)
            with (output/"history.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            control[:]=torch.tensor([ap,best_ap,int(bad_epochs>=config["patience"])],device=device)
            print(json.dumps(row),flush=True)
        if distributed:
            dist.broadcast(control,src=0)
        if control[2].item():
            break
    if rank==0:
        save_json(output/"training_summary.json",{"seconds_this_session":time.perf_counter()-start,
            "completed_epochs":len(history),"best_mAP50_95":best_ap,"world_size":world,
            "global_batch":config["batch"],"seed":config["seed"],
            "resume_note":"Optimizer/scheduler/scaler restored; stochastic worker state is not guaranteed bit-identical after resume."})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--evaluate",action="store_true")
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.evaluate:
        seed_everything(config["seed"])
        model=build_model(False,config["size"])
        checkpoint=torch.load(Path(config["output"])/"best.pt",map_location="cpu",weights_only=False)
        model.load_state_dict(checkpoint["model"])
        final_evaluation(model,"fasterrcnn",config)
    else:
        train(config)


if __name__=="__main__":
    main()
