# CO3.1 Final Project — RF-DETR Hard Hat Detection

Object detection system using RF-DETR (Receptive Field-Enhanced Detection Transformer) to detect safety helmets and exposed heads in construction site images.

---

## Dataset

**Hard Hat Universe** via Roboflow Universe  
- **Classes:** `Hard-Hats`, `head`, `helmet`  
- **Format:** COCO JSON  
- **Split:** 4,878 train / 1,405 val / 707 test images  
- **Source:** https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t

---

## Model

- **Architecture:** RF-DETR Base (~31.9M parameters)
- **Backbone:** DINOv2 Vision Transformer
- **Framework:** PyTorch via the `rfdetr` package
- **Pretrained on:** COCO (fine-tuned on Hard Hat Universe)

---

## Results

| Metric | Value |
|--------|-------|
| mAP@0.50 | 91.09% |
| mAP@0.50:0.95 | 62.42% |
| Precision | 97.42% |
| Recall | 91.23% |
| F1 Score | 94.22% |
| Accuracy | 89.07% |

---

## Setup

### 1. Install dependencies

```bash
pip install rfdetr roboflow supervision pycocotools matplotlib Pillow
pip install "rfdetr[train,loggers]"
```

### 2. Set your Roboflow API Key

In the notebook, replace the placeholder in Cell 3:

```python
API_KEY = "your_roboflow_api_key_here"
```

Get your key at: https://app.roboflow.com → Settings → Roboflow Keys

### 3. Run on Google Colab

1. Upload the `.ipynb` file to Google Colab
2. Set runtime to **T4 GPU** (Runtime > Change runtime type)
3. Run all cells in order

> **Note:** Due to T4 VRAM limits (15GB), training uses `batch_size=1` with `grad_accum_steps=16` to maintain an effective batch size of 16.

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Install all dependencies |
| 2 | Mount Google Drive for checkpointing |
| 3 | Download dataset via Roboflow API |
| 4 | Verify dataset structure and class names |
| 5 | Visualize sample annotated images |
| 6 | Load RF-DETR Base model |
| 7 | Train the model (10 epochs, early stopping) |
| 8 | COCO evaluation — mAP@0.50 and mAP@0.50:0.95 |
| 9 | Per-class Precision, Recall, F1, and Accuracy |
| 10 | Metrics bar chart |
| 11 | Visualize predictions on test images |
| 12 | Save all outputs to Google Drive |

---

## References

- RF-DETR: https://github.com/roboflow/rf-detr
- Roboflow: https://roboflow.com
- DETR Paper: https://arxiv.org/abs/2005.12872
- DINOv2: https://arxiv.org/abs/2304.07193
