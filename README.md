# pcb-yolov8-spd (Stage 1)

Modular training/evaluation/inference code for PCB defect detection with YOLOv8.

Supported model variants:
- **baseline**: official Ultralytics `ultralytics` (pip)
- **spdconv**: Ultralytics fork integrating SPD-Conv (GitHub)

---

## 1) Install

### Baseline environment (recommended)
```bash
pip install -r requirements-baseline.txt
pip install -e .
```

### SPDConv environment (recommended: separate env)
SPDConv uses a fork of Ultralytics that still provides the `ultralytics` python package.
Because of that, it can conflict with baseline installs, so **use a separate venv/conda env**.

Option A (requirements):
```bash
pip install -r requirements-spdconv.txt
pip install -e .
```

Option B (script):
```bash
bash scripts/install_spdconv.sh
pip install -r requirements-spdconv.txt
pip install -e .
```

SPDConv fork repo: Cateners/yolov8-spd. (https://github.com/Cateners/yolov8-spd)

---

## 2) Prepare dataset (VOC ➜ YOLO)

```bash
python scripts/prepare_dataset.py --root /path/to/VOC_ROOT --out data/pcb_yolo --seed 42
```

This generates:
- `data/pcb_yolo/data.yaml`
- `data/pcb_yolo/splits.json` (reproducible train/val/test)

---

## 3) Train

Baseline:
```bash
python scripts/train.py --model baseline --weights yolov8s.pt --data data/pcb_yolo/data.yaml --cfg configs/train_baseline.yaml --project runs_pcb --name yolov8s_baseline
```

SPDConv:
```bash
python scripts/train.py --model spdconv --weights yolov8s.pt --data data/pcb_yolo/data.yaml --cfg configs/train_spdconv.yaml --project runs_pcb --name yolov8s_spdconv
```

---

## 4) Evaluate (val/test)
```bash
python scripts/evaluate.py --model baseline --weights runs_pcb/yolov8s_baseline/weights/best.pt --data data/pcb_yolo/data.yaml --split test
python scripts/evaluate.py --model spdconv --weights runs_pcb/yolov8s_spdconv/weights/best.pt --data data/pcb_yolo/data.yaml --split test
```

---

## 5) Infer a single image
```bash
python scripts/infer.py --model baseline --weights runs_pcb/yolov8s_baseline/weights/best.pt --img /path/to/image.jpg --save_dir outputs/infer
```

Notes:
- wandb is disabled by default to avoid Colab issues.
- PyTorch 2.6+ `weights_only` change is handled for trusted checkpoints.
