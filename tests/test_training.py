"""CPU checks for geometry, negative samples and detection metric edge cases."""
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"training"))
from common import (CocoDataset, PRIMARY_METRICS, coco_metrics, letterbox, operating_metrics,
                    predict_dataset, prepare_gpu_smoke_config, select_threshold, transform_boxes)
from yolov8_train import YOLOCocoDataset


@pytest.fixture
def annotation(tmp_path):
    data={"images":[{"id":1,"width":160,"height":280,"file_name":"positive.png"},
                    {"id":2,"width":160,"height":280,"file_name":"negative.png"}],
          "annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[10,20,30,40],"area":1200,"iscrowd":0}],
          "categories":[{"id":1,"name":"varroa"}],"info":{}}
    path=tmp_path/"instances_val.json";path.write_text(json.dumps(data));return path


def prediction(image_id,score,box=(10,20,30,40)):
    return {"image_id":image_id,"category_id":1,"bbox":list(box),"score":score}


def test_letterbox_round_trip():
    tensor,geo=letterbox(Image.new("RGB",(160,280)),640)
    assert tensor.shape==(3,640,640)
    boxes=torch.tensor([[0.,0.,160.,280.],[10.,20.,40.,60.]])
    recovered=transform_boxes(transform_boxes(boxes,geo),geo,inverse=True)
    assert torch.allclose(boxes,recovered,atol=1e-5)


def test_duplicate_detection_is_false_positive(annotation):
    preds=[prediction(1,.9),prediction(1,.8),prediction(2,.7)]
    m=operating_metrics(annotation,preds,.5)
    assert (m["TP"],m["FP"],m["FN"])==(1,2,0)
    assert m["mean_IoU_TP"]==1 and m["negative_image_false_alarm_rate"]==1
    threshold,_=select_threshold(annotation,preds)
    assert threshold==.9
    assert operating_metrics(annotation,preds,threshold)["F1"]==1


def test_empty_predictions_count_missed_objects(annotation):
    m=operating_metrics(annotation,[],.5)
    assert (m["TP"],m["FP"],m["FN"])==(0,0,1)
    assert m["recall"]==0 and m["mean_IoU_TP"] is None
    assert coco_metrics(annotation,[])["mAP50_95"]==0


def test_perfect_coco_predictions(annotation):
    m=coco_metrics(annotation,[prediction(1,.9)])
    assert m["mAP50_95"]==pytest.approx(1)
    assert m["AP50"]==pytest.approx(1)
    assert m["AP_large"] is None


def test_actual_dataset_negative_and_yolo_batch():
    root=Path(__file__).resolve().parents[1]/"dataset/varroa_ready"
    if not (root / "images/train").is_dir():
        pytest.skip("Prepared local dataset unavailable")
    d=CocoDataset(root,"train",128)
    pos=next(i for i,im in enumerate(d.images) if d.annotations[im["id"]])
    neg=next(i for i,im in enumerate(d.images) if not d.annotations[im["id"]])
    negative=d[neg]
    assert negative[1]["boxes"].shape==(0,4)
    batch=YOLOCocoDataset.collate_fn([negative,d[pos]])
    assert batch["img"].shape==(2,3,128,128)
    assert batch["img"].min()>=0 and batch["img"].max()<=1
    assert (batch["cls"]==0).all() and (batch["batch_idx"]==1).all()
    assert (batch["bboxes"]>=0).all() and (batch["bboxes"]<=1).all()


def test_only_false_positives_selects_rejecting_threshold(annotation):
    threshold,_=select_threshold(annotation,[prediction(2,.9)])
    assert threshold==1.0
    assert operating_metrics(annotation,[prediction(2,.9)],threshold)["FP"]==0


def test_primary_metrics_match_user_csv():
    import csv
    path=Path(__file__).resolve().parents[1]/"docs/Lab3_Model_Evaluation_Metrics.csv"
    with path.open(encoding="utf-8-sig",newline="") as f:
        assert tuple(row["metric_key"] for row in csv.DictReader(f))==PRIMARY_METRICS


def test_spawn_dataloader_and_gpu_smoke_subset(tmp_path):
    root=Path(__file__).resolve().parents[1]/"dataset/varroa_ready"
    if not (root / "images/train").is_dir():
        pytest.skip("Prepared local dataset unavailable")
    config={"root":str(root),"batch":8,"output":"unused","resume":"unused"}
    path=prepare_gpu_smoke_config(config,tmp_path)
    c=json.loads(path.read_text())
    assert c["epochs"]==1 and c["resume"] is None and c["output"]!="unused"
    assert config["resume"]=="unused" and config["root"]==str(root)
    assert not (Path(c["root"])/"annotations/instances_test.json").exists()
    ds=CocoDataset(c["root"],"train",128)
    assert len(ds)==8
    assert sum(bool(ds.annotations[i["id"]]) for i in ds.images)==4
    class EmptyPredictor:
        device=torch.device("cpu")
        def __call__(self,images,geometry,ids):
            return []
    assert predict_dataset(EmptyPredictor(),ds,batch_size=2,workers=2)==[]


def test_final_csv_precision_is_numeric_not_compute_dtype(annotation,tmp_path,monkeypatch):
    import csv
    from types import SimpleNamespace
    import common
    monkeypatch.setattr(common,"CocoDataset",lambda *a,**k:SimpleNamespace(annotation_path=annotation))
    monkeypatch.setattr(common,"Predictor",lambda *a,**k:object())
    monkeypatch.setattr(common,"predict_dataset",lambda *a,**k:[prediction(1,.9)])
    # Deliberately include the old colliding metadata key to protect CSV semantics.
    monkeypatch.setattr(common,"benchmark",lambda *a,**k:{"FPS":10.0,"precision":"FP32"})
    monkeypatch.setattr(common,"render_examples",lambda *a,**k:None)
    config=dict(output=str(tmp_path),root="unused",eval_device="cpu",size=128,
                eval_batch=2,workers=0,seed=42,benchmark_samples=2)
    metrics=common.final_evaluation(torch.nn.Linear(1,1),"test",config)
    with (tmp_path/"comparison.csv").open(newline="") as f:
        reader=csv.DictReader(f);row=next(reader)
    assert tuple(row)==("model","seed",*PRIMARY_METRICS)
    assert float(row["precision"])==metrics["test"]["precision"]==1.0
    assert float(row["FPS"])==10.0
