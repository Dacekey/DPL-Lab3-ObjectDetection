"""Regenerate paper figures and exact tables from stored experiment artifacts."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
FIG = ROOT / "figures"
GEN = ROOT / "generated"
RUNS = {"Faster R-CNN": LAB / "output/faster_r-cnn/fasterrcnn_seed42",
        "YOLOv8s": LAB / "output/yolo/yolov8s_seed42"}
COLORS = ["#1f619d", "#df7f24"]


def main():
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    metrics = {k: json.loads((v / "metrics.json").read_text()) for k, v in RUNS.items()}
    histories = {}
    for name, path in RUNS.items():
        file = "history.csv" if name == "Faster R-CNN" else "results.csv"
        histories[name] = [{k: float(v) for k, v in r.items()}
                           for r in csv.DictReader((path / file).open())]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.9))
    for i, (name, rows) in enumerate(histories.items()):
        ap = "val_mAP50_95" if i == 0 else "val/COCO_mAP50_95"
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r[ap]*100 for r in rows], label=name, color=COLORS[i])
        best = max(rows, key=lambda r: r[ap])
        axes[0].scatter(best["epoch"], best[ap]*100, color=COLORS[i], marker="*", s=90)
        if i == 0:
            axes[1].plot(epochs, [r["train_loss"] for r in rows], color=COLORS[i])
        else:
            for key, style in [("train/box_loss", "-"), ("train/cls_loss", "--"), ("train/dfl_loss", ":")]:
                axes[2].plot(epochs, [r[key] for r in rows], style, label=key.split("/")[1])
    axes[0].set(title="Validation AP (best epoch: star)", ylabel="AP@[.50:.95] (%)")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Faster R-CNN: total train loss", ylabel="Loss")
    axes[2].set(title="YOLOv8s: train loss components", ylabel="Loss")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG / "training.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    fields = ["mAP50_95", "AP50", "AP75", "precision", "recall", "F1"]
    x = np.arange(len(fields))
    for i, (name, m) in enumerate(metrics.items()):
        axes[0].bar(x+(i-.5)*.36, [m["test"][k]*100 for k in fields], .36,
                    label=name, color=COLORS[i])
        axes[1].scatter(m["benchmark"]["FPS"], m["test"]["mAP50_95"]*100,
                        s=m["parameters"]/1e5, color=COLORS[i], alpha=.8)
        axes[1].annotate(name, (m["benchmark"]["FPS"], m["test"]["mAP50_95"]*100),
                         xytext=(0, 15), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_xticks(x, ["AP", "AP50", "AP75", "P", "R", "F1"])
    axes[0].set(ylabel="Test score (%)", ylim=(0, 100))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="FPS (one T4, FP32, batch 1)", ylabel="Test AP@[.50:.95] (%)",
                xlim=(0, 65), ylim=(15, 36), title="Bubble area proportional to parameter count")
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for i, (name, path) in enumerate(RUNS.items()):
        curve = json.loads((path / "validation_PR_F1_curve.json").read_text())
        axes[0].plot([r["recall"] for r in curve], [r["precision"] for r in curve],
                     label=name, color=COLORS[i])
        axes[1].plot([r["threshold"] for r in curve], [r["f1"] for r in curve],
                     label=name, color=COLORS[i])
        m = metrics[name]["validation"]
        axes[1].scatter(m["confidence_threshold"], m["F1"], color=COLORS[i], marker="*", s=90)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Validation operating curve (IoU=0.5)")
    axes[1].set(xlabel="Confidence threshold", ylabel="F1", title="Validation threshold selection")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "validation_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    table = [r"\begin{tabular}{lrr}", r"\toprule", r"Chỉ số test & Faster R-CNN & YOLOv8s \\", r"\midrule"]
    labels = {"mAP50_95": r"AP@[.50:.95] (\%)", "AP50": r"AP50 (\%)", "AP75": r"AP75 (\%)",
              "precision": r"Precision (\%)", "recall": r"Recall (\%)", "F1": r"F1 (\%)",
              "mean_IoU_TP": r"Mean IoU TP (\%)", "AP_small": r"AP small (\%)", "AP_medium": r"AP medium (\%)"}
    for k, label in labels.items():
        table.append(label + " & " + " & ".join(f"{metrics[n]['test'][k]*100:.2f}".replace(".", ",") for n in RUNS) + r" \\")
    for k in ("TP", "FP", "FN"):
        table.append(k + " & " + " & ".join(str(metrics[n]["test"][k]) for n in RUNS) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (GEN / "test_table.tex").write_text("\n".join(table)+"\n")
    provenance = {str(p.relative_to(LAB)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for run in RUNS.values() for p in run.iterdir()
                  if p.suffix in {".json", ".csv"}}
    (GEN / "artifact_sha256.json").write_text(json.dumps(provenance, indent=2)+"\n")
    write_qualitative(metrics)
    print("Figures, table and provenance generated:", ROOT)


def write_qualitative(metrics):
    data = json.loads((LAB / "dataset/varroa_ready/annotations/instances_test.json").read_text())
    truth = defaultdict(list)
    preds = {name: defaultdict(list) for name in RUNS}
    for a in data["annotations"]: truth[a["image_id"]].append(a["bbox"])
    for name, path in RUNS.items():
        for p in json.loads((path / "test_predictions.json").read_text()):
            if p["score"] >= metrics[name]["test"]["confidence_threshold"]:
                preds[name][p["image_id"]].append(p)
    def matched(i, name):
        used=set()
        for p in sorted(preds[name][i], key=lambda p:-p["score"]):
            x,y,w,h=p["bbox"]; scores=[]
            for j,(a,b,c,d) in enumerate(truth[i]):
                inter=max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
                scores.append(-1 if j in used else inter/(w*h+c*d-inter))
            if scores and max(scores)>=.5:used.add(max(range(len(scores)),key=scores.__getitem__))
        return len(used)
    cases={}
    for im in data["images"]:
        i=im["id"];f=matched(i,"Faster R-CNN");y=matched(i,"YOLOv8s")
        if truth[i] and f>y:cases.setdefault("Faster R-CNN recovers a missed object",im)
        if not truth[i] and preds['Faster R-CNN'][i] and not preds['YOLOv8s'][i]:
            cases.setdefault("False alarm on an annotated negative",im)
        if truth[i] and f==0 and y==0:cases.setdefault("Both models miss the annotated object",im)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",11)
    selected=[]
    for label in ["Faster R-CNN recovers a missed object",
                  "False alarm on an annotated negative",
                  "Both models miss the annotated object"]:
        if label not in cases: continue
        im = cases[label]
        source=LAB/"dataset/varroa_ready"/im["file_name"]
        if not source.is_file():continue
        sheet=Image.new("RGB",(480,320),"white")
        for col,name in enumerate(["Ground truth","Faster R-CNN","YOLOv8s"]):
            with Image.open(source) as image:panel=image.convert("RGB")
            draw=ImageDraw.Draw(panel)
            boxes=truth[im['id']] if col==0 else [p['bbox'] for p in preds[name][im['id']]]
            for x,y,w,h in boxes:draw.rectangle((x,y,x+w,y+h),outline=("lime" if col==0 else "red"),width=2)
            sheet.paste(panel,(col*160,40))
            ImageDraw.Draw(sheet).text((col*160+3,23),name,font=font,fill="black")
        ImageDraw.Draw(sheet).text((3,3),f"ID {im['id']}: {label}",font=font,fill="black")
        target=f"qualitative_{len(selected)+1}.png";sheet.save(FIG/target)
        selected.append({"figure":target,"image_id":im["id"],"file_name":im["file_name"],"case":label,
                         "selection":"first image satisfying the case in original test image order"})
    (GEN/"qualitative_manifest.json").write_text(json.dumps(selected,indent=2)+"\n")


if __name__ == "__main__":
    main()
