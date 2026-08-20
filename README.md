# 🧠 CerebraAI

Intracranial Hemorrhage Detection, Classification & Segmentation — a research/educational prototype built with Streamlit.

Upload a head CT slice and CerebraAI will:
1. **Detect** candidate hemorrhage regions (YOLOv11)
2. **Segment** each region with a precise mask (MobileSAM, box-prompted)
3. **Classify** the subtype (ResNet18): Epidural, Intraparenchymal, Intraventricular, or Subdural

> ⚠️ **Not for clinical use.** This is a research/educational prototype only and must never be used to diagnose or guide treatment of real patients.

---

## How it works

| Stage | Model | File |
|---|---|---|
| Detection | YOLOv11n, trained on the Hssayeni CT-ICH dataset | `finalmodel.pt` |
| Segmentation | MobileSAM, prompted with the detector's boxes | `mobile_sam.pt` |
| Classification | ResNet18, 4 subtypes | `classifier_best.pt` |

For each detected region, the app pads and crops around the box, classifies the crop, and overlays a color-coded, contoured mask on the original image along with detection and classification confidence. Subarachnoid hemorrhage is intentionally excluded from the classifier due to insufficient training data.

Only `app.py` is used to run the app. The repo also contains `app2.py`, `modelv0.pt`, and `modelv2.pt` from earlier iterations — these are not required and can be ignored (or removed) for this version.

---

## Requirements

- Python 3.10+
- The following packages:

```
streamlit
numpy
opencv-python
torch
torchvision
ultralytics
pillow
```

(`requirements.txt` in this repo is a full environment export and includes many unrelated packages; the list above is what `app.py` actually needs.)

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install streamlit numpy opencv-python torch torchvision ultralytics pillow

# 3. Make sure these model weights are present in the project root:
#    finalmodel.pt, mobile_sam.pt, classifier_best.pt
```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), upload a CT slice (`.jpg`, `.jpeg`, `.png`) from the sidebar, and adjust:

- **Detection confidence** — minimum confidence for the detector to flag a region
- **Mask opacity** — how strongly the segmentation overlay is blended onto the image
- **Show raw detection box** — toggle the coarse YOLO box on/off (the mask is the precise output)

## Project structure

```
.
├── app.py                 # Streamlit app (the only entry point used)
├── finalmodel.pt           # YOLO detector weights
├── mobile_sam.pt            # MobileSAM segmentation weights
├── classifier_best.pt       # ResNet18 subtype classifier weights
└── requirements.txt         # Full environment export
```
## 🎥 Demo

![Demo](demoo.gif)

## 👥 Team Members:
Yousef Sabri
Magdy Hassaan
Omar Bassem
Amir 
