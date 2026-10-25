"""YOLOv8s with the same COCO Dataset adapter and evaluator as Faster R-CNN.

Pin ultralytics==8.3.228: the trainer extension is tested against that implementation.
Launch explicitly with torchrun so the custom subclass is imported on every rank.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from common import (CocoDataset, Predictor, coco_metrics, final_evaluation,
                    predict_dataset, save_json, seed_everything, seed_worker)


class YOLOCocoDataset(CocoDataset):
    """Keep the input pipeline identical; adapt targets only at the model boundary."""
    @staticmethod
    def collate_fn(batch):
        images,targets,geometry=zip(*batch)
        bboxes,classes,indices=[],[],[]
        for i,(image,target) in enumerate(zip(images,targets)):
            xyxy=target["boxes"]
            xywh=torch.cat(((xyxy[:,:2]+xyxy[:,2:])/2,xyxy[:,2:]-xyxy[:,:2]),dim=1)/image.shape[-1]
            bboxes.append(xywh)
            classes.append(torch.zeros((len(xywh),1),dtype=torch.float32))
            indices.append(torch.full((len(xywh),),i,dtype=torch.float32))
        return {"img":torch.stack(images),"bboxes":torch.cat(bboxes),
                "cls":torch.cat(classes),"batch_idx":torch.cat(indices),
                "im_file":[str(int(t["image_id"])) for t in targets],
                "ori_shape":[(g[5],g[4]) for g in geometry],
                "resized_shape":[tuple(image.shape[-2:]) for image in images]}


class LabTrainer(DetectionTrainer):
    def __init__(self,*args,**kwargs):
        self.lab_config=json.loads(Path(os.environ["LAB_CONFIG"]).read_text())
        super().__init__(*args,**kwargs)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
        c=self.lab_config
        dataset=YOLOCocoDataset(c["root"],mode,c["size"],augment=mode=="train")
        world=int(os.environ.get("WORLD_SIZE","1"))
        global_rank=int(os.environ.get("RANK","0"))
        sampler=DistributedSampler(dataset,num_replicas=world,rank=global_rank,
            shuffle=mode=="train",seed=c["seed"]) if world>1 else None
        return DataLoader(dataset,batch_size=batch_size,shuffle=mode=="train" and sampler is None,
            sampler=sampler,num_workers=c["workers"],pin_memory=True,
            multiprocessing_context="spawn" if c["workers"] else None,
            collate_fn=YOLOCocoDataset.collate_fn,worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(c["seed"]+global_rank))

    def check_resume(self,overrides):
        super().check_resume(overrides)
        if self.resume:
            c=self.lab_config
            for key,expected in (("seed",c["seed"]),("imgsz",c["size"]),
                                 ("batch",c["batch"]),("epochs",c["epochs"])):
                if getattr(self.args,key)!=expected:
                    raise ValueError(f"Resume requires unchanged {key}")
            # Native resume otherwise retains paths stored in the old Kaggle session.
            self.args.data=c["data_yaml"]
            self.args.project=str(Path(c["output"]).parent)
            self.args.name=Path(c["output"]).name
            self.args.save_dir=c["output"]
            if not (Path(c["output"])/"weights/best.pt").exists():
                raise ValueError("Copy the previous weights/best.pt into the new output before resuming.")

    def preprocess_batch(self,batch):
        # Shared Dataset already supplies float RGB in [0,1], do not divide again.
        return {k:v.to(self.device,non_blocking=True) if isinstance(v,torch.Tensor) else v
                for k,v in batch.items()}

    def validate(self):
        c=self.lab_config
        rank=int(os.environ.get("RANK","0"))
        world=int(os.environ.get("WORLD_SIZE","1"))
        ap=torch.zeros(3,device=self.device)
        if rank==0:
            dataset=CocoDataset(c["root"],"val",c["size"])
            model=self.ema.ema if self.ema else self.model
            predictions=predict_dataset(Predictor(model,"yolov8s",self.device),dataset,
                c["eval_batch"],c["workers"])
            metrics=coco_metrics(dataset.annotation_path,predictions)
            ap[:]=torch.tensor([metrics["mAP50_95"],metrics["AP50"],metrics["AP75"]],device=self.device)
            save_json(Path(c["output"])/"last_validation_metrics.json",metrics)
        if world>1:
            dist.broadcast(ap,src=0)
        fitness=float(ap[0])
        if self.best_fitness is None or fitness>self.best_fitness:
            self.best_fitness=fitness
        return {"val/COCO_mAP50_95":fitness,"val/COCO_AP50":float(ap[1]),
                "val/COCO_AP75":float(ap[2])},fitness

    def final_eval(self):
        # Preserve full last.pt for resume. A separate one-GPU process evaluates best.pt.
        # Native final_eval would invoke a different evaluator and strip optimizer state.
        pass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--evaluate",action="store_true")
    args=parser.parse_args()
    c=json.loads(Path(args.config).read_text())
    os.environ["LAB_CONFIG"]=str(Path(args.config).resolve())
    if args.evaluate:
        seed_everything(c["seed"])
        model=YOLO(str(Path(c["output"])/"weights/best.pt")).model
        final_evaluation(model,"yolov8s",c)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU T4 x2 in Kaggle settings.")
    start=time.perf_counter()
    overrides=dict(model=c["weights"],data=c["data_yaml"],epochs=c["epochs"],
        imgsz=c["size"],batch=c["batch"],device="0,1" if c["gpus"]==2 else "0",
        seed=c["seed"],deterministic=True,workers=c["workers"],amp=c["amp"],
        optimizer="SGD",lr0=c["learning_rate"],lrf=0.01,momentum=0.9,weight_decay=0.0005,
        cos_lr=True,warmup_epochs=c["warmup_epochs"],nbs=c["batch"],
        patience=c["patience"],project=str(Path(c["output"]).parent),
        name=Path(c["output"]).name,exist_ok=True,plots=False,save=True,save_period=10,
        val=True,rect=False,multi_scale=False,mosaic=0.0,mixup=0.0,copy_paste=0.0,
        close_mosaic=0,translate=0.0,scale=0.0,degrees=0.0,shear=0.0,perspective=0.0,
        hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,fliplr=0.0,flipud=0.0,cache=False,
        fraction=1.0,verbose=True,resume=c.get("resume") or False)
    trainer=LabTrainer(overrides=overrides)
    trainer.train()
    if int(os.environ.get("RANK","0"))==0:
        save_json(Path(c["output"])/"training_summary.json",{
            "seconds_this_session":time.perf_counter()-start,"best_mAP50_95":trainer.best_fitness,
            "completed_epochs":trainer.epoch+1,"world_size":c["gpus"],
            "global_batch":c["batch"],"seed":c["seed"],"EMA":True,
            "resume_note":"Native Ultralytics resume; stochastic worker state may differ after resume."})


if __name__=="__main__":
    main()
