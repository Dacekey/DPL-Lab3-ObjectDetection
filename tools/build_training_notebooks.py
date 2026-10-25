"""Generate two self-contained notebooks; no repository upload is needed on Kaggle."""
import json
from pathlib import Path

LAB=Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type":"markdown","metadata":{},"source":source.splitlines(True)}


def code(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(True)}


def build(family,title,filename,learning_rate):
    yolo=family=="yolov8s"
    trainer="yolov8_train.py" if yolo else "fasterrcnn_train.py"
    out_name="yolov8s_seed42" if yolo else "fasterrcnn_seed42"
    packages=["pycocotools==2.0.11"]+(["ultralytics==8.3.228"] if yolo else [])
    cells=[markdown(f'''# Lab 3 — {title} trên Kaggle (2 × T4)

Notebook độc lập: chứa toàn bộ implementation, Dataset adapter và evaluator.
**Không cần upload thư mục code hoặc notebook còn lại.**

1. Tạo Kaggle Notebook, chọn **GPU T4 ×2**, bật **Internet**.
2. Attach Kaggle Dataset đã tạo từ `varroa_kaggle_ready.zip`.
3. Import notebook này rồi **Run All**. Đường dẫn dataset được tìm tự động.
4. Tải ZIP kết quả trong `/kaggle/working/` sau khi hoàn tất.

Notebook tự chạy thử một batch train và 4 ảnh validation trên đúng GPU/AMP/worker
đã chọn trước khi train đầy đủ. Lượt thử dùng output riêng, không đọc test và không
thay đổi weights khởi tạo của run chính. Nếu thử thất bại, notebook dừng tại lỗi đó.

Model: COCO-pretrained, fine-tune một class `varroa`; **seed 42**, tối đa 100 epochs,
early stopping 15 epochs không cải thiện COCO mAP@50:95. Global batch 8 = 4 ảnh/GPU.
Hai notebook dùng cùng code preprocessing: RGB, letterbox 640 × 640 giữ tỷ lệ,
padding 114, lật ngang p=0.5 và brightness/contrast/color trong [0.9,1.1].
Không mosaic, crop hay multi-scale. Validation/test không augmentation.

Chọn checkpoint theo **pycocotools COCO AP@[.50:.05:.95]** trên validation;
chọn confidence theo F1 validation tại IoU=.5 rồi khóa khi chạy test.
Tốc độ đo riêng trên **một T4, FP32, batch 1**, gồm preprocess, chuyển GPU,
model, NMS và chuyển box về ảnh gốc; không gồm đọc file/giải mã ảnh.

{'YOLO dùng EMA của Ultralytics; Faster R-CNN dùng weights trực tiếp. Đây là khác biệt recipe được ghi lại, không phải phép cô lập kiến trúc tuyệt đối.' if yolo else 'Faster R-CNN dùng FrozenBatchNorm và normalization nội bộ của pretrained weights. Ảnh đã letterbox vuông nên cấu hình min_size=max_size=640 để không phóng ảnh lần nữa.'}

Internet cần cho cài thư viện và tải pretrained weights lần đầu. Resume giữ trạng thái
optimizer/scheduler/scaler theo framework, không cam kết giống bit khi worker khởi động lại.
Seed và cuDNN được cố định; không ép mọi kernel deterministic để giữ CUDA ROIAlign
chuẩn của Torchvision. Một số phép toán detection trên GPU vẫn có sai khác nhỏ.
Một seed không cho phép kết luận về độ biến thiên giữa các lần huấn luyện.
'''),code(f'''# 1. Cấu hình — chỉ đổi nếu bạn muốn thay ngân sách training.
from pathlib import Path
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "DISABLED"
os.environ["YOLO_CONFIG_DIR"] = "/kaggle/working/.ultralytics"

DATASET_ROOT = None  # Tự tìm dưới /kaggle/input; hoặc đặt đường dẫn dataset root.
EPOCHS = 100
PATIENCE = 15
GLOBAL_BATCH = 8
GPUS = 2
WORKERS = 2  # Mỗi GPU; giảm về 0 nếu dataloader gặp lỗi shared memory.
EVAL_BATCH = 4
SEED = 42
IMAGE_SIZE = 640
AMP = {yolo!r}  # Faster R-CNN: FP32 để tránh loss NaN/Inf ở FP16; YOLO giữ AMP.
RESUME_CHECKPOINT = None  # Đường dẫn last.pt của một run CHƯA train xong.
OUTPUT = Path("/kaggle/working/{out_name}")
RUNTIME = Path("/kaggle/working/lab3_{family}_runtime")
RUNTIME.mkdir(parents=True,exist_ok=True)
OUTPUT.mkdir(parents=True,exist_ok=True)
if not RESUME_CHECKPOINT and any((OUTPUT/name).exists() for name in ("best.pt","last.pt","weights/best.pt","weights/last.pt")):
    raise RuntimeError("Output đã có checkpoint. Chọn OUTPUT mới hoặc đặt RESUME_CHECKPOINT=last.pt.")
'''),code(f'''# 2. Cài các dependency bổ sung; giữ cặp torch/torchvision sẵn có của Kaggle.
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","--quiet",{','.join(repr(p) for p in packages)}],check=True)
import torch, torchvision
assert torch.cuda.is_available(), "Chọn GPU T4 x2 trong Kaggle Settings."
assert torch.cuda.device_count() >= GPUS, "Notebook yêu cầu đủ 2 GPU; kiểm tra accelerator."
assert GLOBAL_BATCH % GPUS == 0
print("torch:",torch.__version__,"torchvision:",torchvision.__version__)
for index in range(GPUS):
    print(index,torch.cuda.get_device_name(index))
'''),code('''# 3. Tìm dataset và ghi cấu hình vào vùng có quyền ghi.
import json, shutil, zipfile, hashlib, importlib.metadata

def find_dataset():
    if DATASET_ROOT:
        root=Path(DATASET_ROOT)
        assert (root/"annotations/instances_train.json").exists()
        return root.resolve()
    candidates=sorted({p.parent.parent for p in Path("/kaggle/input").rglob("instances_train.json")
                       if p.parent.name=="annotations" and (p.parent.parent/"data.yaml").exists()})
    if not candidates:
        archives=list(Path("/kaggle/input").rglob("varroa_kaggle_ready.zip"))
        assert len(archives)==1, "Attach varroa_kaggle_ready dataset hoặc đặt DATASET_ROOT."
        root=Path("/kaggle/working/varroa_data")
        root.mkdir(exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.namelist():
                assert (root/member).resolve().is_relative_to(root.resolve()), "Unsafe ZIP path"
            archive.extractall(root)
        candidates=[root]
    assert len(candidates)==1, f"Tìm thấy {len(candidates)} dataset; đặt DATASET_ROOT để chọn."
    return candidates[0].resolve()

root=find_dataset()
for split in ("train","val","test"):
    data=json.loads((root/"annotations"/f"instances_{split}.json").read_text())
    assert data["categories"]==[{"id":1,"name":"varroa","supercategory":"mite"}]
    assert all((root/im["file_name"]).is_file() for im in data["images"])
    print(split,len(data["images"]),"images;",len(data["annotations"]),"boxes")
runtime_yaml=RUNTIME/"data.yaml"
runtime_yaml.write_text(f'path: {json.dumps(str(root))}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\nnc: 1\\nnames: [varroa]\\n')
'''),code(f'''CONFIG = dict(root=str(root),data_yaml=str(runtime_yaml),output=str(OUTPUT),
    epochs=EPOCHS,patience=PATIENCE,batch=GLOBAL_BATCH,gpus=GPUS,workers=WORKERS,
    seed=SEED,size=IMAGE_SIZE,amp=AMP,learning_rate={learning_rate},warmup_epochs=3,
    eval_batch=EVAL_BATCH,eval_device="cuda:0",benchmark_samples=100,
    resume=RESUME_CHECKPOINT,weights=str(RUNTIME/"yolov8s.pt"))
CONFIG_PATH=RUNTIME/"config.json"
CONFIG_PATH.write_text(json.dumps(CONFIG,indent=2))
(OUTPUT/"config.json").write_text(json.dumps(CONFIG,indent=2))
environment=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,
    CUDA=torch.version.cuda,GPUs=[torch.cuda.get_device_name(i) for i in range(GPUS)],
    packages={{p:importlib.metadata.version(p) for p in {['pycocotools','ultralytics'] if yolo else ['pycocotools']!r}}},
    annotation_sha256={{s:hashlib.sha256((root/"annotations"/f"instances_{{s}}.json").read_bytes()).hexdigest()
                       for s in ("train","val","test")}},
    evaluator="pycocotools",score_floor=0.001,NMS_IoU=0.5,max_detections=100,
    precision_eval="FP32",EMA={yolo!r},native_detection_kernels=True,
    normalization={('RGB / 255' if yolo else 'RGB / 255 plus pretrained Torchvision internal mean/std')!r})
(OUTPUT/"environment.json").write_text(json.dumps(environment,indent=2))
(OUTPUT/"requirements_freeze.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
print("Dataset:",root,"\\nOutput:",OUTPUT)
'''),markdown('''## Implementation dùng chung

Cell tiếp theo ghi module vào `/kaggle/working/`. Đó là mã được đóng gói ngay trong
notebook, không cần tìm file trên máy local. YOLO và Faster R-CNN dùng **cùng module này**.
COCO AP giữ score floor .001; P/R/F1 dùng ngưỡng được chọn riêng trên validation.
Mean IoU chỉ tính trên TP ghép một-một ở IoU≥.5; đọc cùng Recall.
AP/AR của nhóm không có ground truth được xuất `null`, không coi là điểm 0.
'''),code(f'%%writefile /kaggle/working/lab3_{family}_runtime/common.py\n'+(LAB/"training/common.py").read_text()),
markdown(f'## Implementation {title}\n\n'+('Subclass trainer Ultralytics dùng COCO Dataset chung, loss/head YOLOv8s gốc, và COCO validation chung làm fitness để lưu best.pt. Khởi chạy bằng torchrun để mỗi rank dùng đúng subclass.' if yolo else 'Thay FastRCNNPredictor thành 2 class (background + Varroa); training loop DDP, FP32, SGD, warmup/cosine LR, validation và checkpoint đầy đủ. FP32 là mặc định sau lỗi loss NaN/Inf ở run AMP; nguyên nhân số học cụ thể chưa được xác nhận. Nếu loss/gradient không hữu hạn, mỗi rank lưu nonfinite_rank*.json với loss, learning rate và image ID/box trước khi dừng cả hai GPU.')),
code(f'%%writefile /kaggle/working/lab3_{family}_runtime/{trainer}\n'+(LAB/"training"/trainer).read_text()),
code('''# Resume từ output cũ đã attach: giữ best checkpoint và history của run trước.
if RESUME_CHECKPOINT:
    previous_last=Path(RESUME_CHECKPOINT)
    assert previous_last.is_file(), "Không tìm thấy last.pt để resume."
    previous_best=previous_last.parent/"best.pt"
    assert previous_best.is_file(), "Cần giữ cả best.pt và last.pt từ run trước."
    is_yolo=previous_last.parent.name=="weights"
    destination=OUTPUT/"weights/best.pt" if is_yolo else OUTPUT/"best.pt"
    destination.parent.mkdir(parents=True,exist_ok=True)
    if previous_best.resolve()!=destination.resolve():
        shutil.copy2(previous_best,destination)
    old_output=previous_last.parent.parent if is_yolo else previous_last.parent
    for name in ("results.csv","history.csv"):
        source=old_output/name
        if source.exists() and source.resolve()!=(OUTPUT/name).resolve():
            shutil.copy2(source,OUTPUT/name)
'''+('''# 4. Tải pretrained YOLOv8s một lần trước khi spawn hai process.
from ultralytics import YOLO
download=YOLO("yolov8s.pt")
source=Path("yolov8s.pt").resolve()
if source != (RUNTIME/"yolov8s.pt").resolve():
    shutil.copy2(source,RUNTIME/"yolov8s.pt")
del download
''' if yolo else '''# 4. Tải pretrained COCO một lần trước khi spawn hai process.
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
download=FasterRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True,check_hash=True)
del download
''')+'''
# Chỉ lưu mã và config, tránh copy theo symlink ảnh của GPU smoke test khi resume.
source_output=OUTPUT/"source"
source_output.mkdir(parents=True,exist_ok=True)
for source in RUNTIME.iterdir():
    if source.is_file() and source.suffix in (".py",".json",".yaml"):
        shutil.copy2(source,source_output/source.name)

def run_command(command,env=None):
    print(" ".join(map(str,command)),flush=True)
    process=subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    for line in process.stdout:
        print(line,end="",flush=True)
    status=process.wait()
    if status:
        raise subprocess.CalledProcessError(status,command)
'''),code(f'''# 5a. Chạy thử GPU/DDP/AMP và workers trước khi bắt đầu run chính.
sys.path.insert(0,str(RUNTIME))
from common import prepare_gpu_smoke_config
smoke_config=prepare_gpu_smoke_config(CONFIG,RUNTIME)
run_command([sys.executable,"-m","torch.distributed.run","--standalone",
    f"--nproc_per_node={{GPUS}}",RUNTIME/{trainer!r},"--config",smoke_config])
print("GPU preflight passed. Bắt đầu training đầy đủ ở cell tiếp theo.")
'''),code(f'''# 5b. Huấn luyện trên 2 GPU. Nếu OOM: giảm GLOBAL_BATCH và EVAL_BATCH,
# rồi chạy lại từ đầu phần cấu hình; không tự đổi batch ở giữa thí nghiệm.
if not RESUME_CHECKPOINT and any((OUTPUT/name).exists() for name in ("best.pt","last.pt","weights/best.pt","weights/last.pt")):
    raise RuntimeError("Output đã có checkpoint. Chọn OUTPUT mới hoặc đặt RESUME_CHECKPOINT=last.pt.")
run_command([sys.executable,"-m","torch.distributed.run","--standalone",
    f"--nproc_per_node={{GPUS}}",RUNTIME/{trainer!r},"--config",CONFIG_PATH])
'''),code(f'''# 6. Đánh giá best checkpoint trong process riêng, chỉ GPU 0.
# Chạy cell này sau khi training kết thúc; có thể chạy lại mà không train lại.
evaluation_env=os.environ.copy()
evaluation_env["CUDA_VISIBLE_DEVICES"]="0"
run_command([sys.executable,RUNTIME/{trainer!r},"--config",CONFIG_PATH,"--evaluate"],env=evaluation_env)
'''),code('''# 7. Kết quả chính và hình minh họa (xanh: ground truth; đỏ: prediction).
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
display(pd.read_csv(OUTPUT/"comparison.csv").T)
display(Image.open(OUTPUT/"test_examples.jpg"))
curve=pd.DataFrame(json.loads((OUTPUT/"validation_PR_F1_curve.json").read_text()))
if not curve.empty:
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(curve["recall"],curve["precision"])
    axes[0].set(xlabel="Recall",ylabel="Precision",title="Validation PR (IoU=0.5)",xlim=(0,1),ylim=(0,1))
    axes[1].plot(curve["threshold"],curve["f1"])
    axes[1].set(xlabel="Confidence",ylabel="F1",title="Validation F1")
    fig.tight_layout();fig.savefig(OUTPUT/"validation_curves.png",dpi=150);plt.show()
history_path=OUTPUT/("results.csv" if (OUTPUT/"results.csv").exists() else "history.csv")
history=pd.read_csv(history_path)
history.columns=history.columns.str.strip()
ap_column="val/COCO_mAP50_95" if "val/COCO_mAP50_95" in history else "val_mAP50_95"
if ap_column in history:
    history.plot(x="epoch",y=ap_column,title="Validation COCO mAP@50:95")
    plt.savefig(OUTPUT/"training_AP.png",dpi=150,bbox_inches="tight");plt.show()
'''),code('''# 8. Gói toàn bộ kết quả để tải xuống từ Kaggle Output.
archive=shutil.make_archive(str(OUTPUT)+"_artifacts","zip",root_dir=OUTPUT)
print("Download:",archive)
print("Để so sánh hai model: ghép comparison.csv từ hai run.")
'''),markdown('''## Artifact được lưu

- `best.pt` / `weights/best.pt`: checkpoint chọn bằng COCO mAP@50:95 validation.
- `last.pt` / `weights/last.pt`: checkpoint có optimizer để resume trước khi train xong.
- `history.csv` / `results.csv`: loss và validation AP từng epoch.
- `comparison.csv`: đúng 7 metric chính: mAP50:95, AP50, Precision, Recall, F1,
  Mean IoU trên TP và FPS; kèm tên model và seed.
- `metrics.json`: kết quả chi tiết bổ sung để tra cứu khi cần.
- `confidence_selection.json`: ngưỡng chọn trên validation.
- `validation_predictions.json`, `test_predictions.json`: COCO predictions theo tọa độ ảnh gốc.
- `config.json`, `environment.json`, `requirements_freeze.txt`, `source/`: cấu hình và mã của run.
- `test_examples.jpg`, `validation_curves.png`, `training_AP.png`: hình cho báo cáo.

Latency giữa hai run chỉ so sánh khi cùng loại T4, batch 1, FP32 và phạm vi đo.
Không so sánh trực tiếp trị số loss của hai kiến trúc. COCO pretraining và recipe riêng
được công bố; kết quả là so sánh hai hệ thống fine-tune, không chỉ riêng kiến trúc.

Tài liệu: [Torchvision Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html),
[Ultralytics training](https://docs.ultralytics.com/modes/train/),
[COCO evaluation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
''')]
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11"},
        "kaggle":{"accelerator":"nvidiaTeslaT4","isInternetEnabled":True,"language":"python","sourceType":"notebook","isGpuEnabled":True}},
        "nbformat":4,"nbformat_minor":5}
    for i,cell in enumerate(cells):
        cell["id"]=f"{family}-{i:02d}"
    path=LAB/"notebooks"/filename
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
    print(path)


if __name__=="__main__":
    build("fasterrcnn","Faster R-CNN ResNet50-FPN","train_fasterrcnn_kaggle.ipynb",0.005)
    build("yolov8s","YOLOv8s","train_yolov8s_kaggle.ipynb",0.01)
