import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn

from torchvision import models, transforms
from ultralytics import YOLO, SAM
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        background: linear-gradient(135deg, #1e2749 0%, #0e1117 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #2a3352;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #4f8cff, #6ee7d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #8b93b0;
        margin-top: 0.3rem;
        font-size: 0.95rem;
    }
    .info-card {
        background: #161b2e;
        border: 1px solid #2a3352;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .detection-card {
        background: #161b2e;
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--accent-color);
    }
    .detection-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f0f2f8;
        margin-bottom: 0.2rem;
    }
    .detection-meta {
        color: #8b93b0;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
    }
    .conf-bar-bg {
        background: #2a3352;
        border-radius: 6px;
        height: 8px;
        margin: 4px 0 10px 0;
    }
    .conf-bar-fill {
        height: 8px;
        border-radius: 6px;
    }
    .stat-box {
        background: #161b2e;
        border: 1px solid #2a3352;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f2f8;
    }
    .stat-label {
        color: #8b93b0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SUBTYPE METADATA
# ============================================================

SUBTYPE_INFO = {
    "Epidural": {
        "color": "#ff5c5c",
        "icon": "🔴",
        "blurb": "Bleeding between the skull and the dura mater, often from arterial injury. Classic biconvex (lens-shaped) appearance on CT.",
        "reliability": "high"
    },
    "Intraparenchymal": {
        "color": "#4f8cff",
        "icon": "🔵",
        "blurb": "Bleeding directly within brain tissue, commonly linked to hypertension or trauma. Appears as a rounded hyperdense region.",
        "reliability": "moderate"
    },
    "Intraventricular": {
        "color": "#ffd23f",
        "icon": "🟡",
        "blurb": "Bleeding into the brain's ventricular system, often extending from a nearby hemorrhage. Frequently associated with hydrocephalus risk.",
        "reliability": "low"
    },
    "Subdural": {
        "color": "#c76eff",
        "icon": "🟣",
        "blurb": "Bleeding between the dura and arachnoid, typically venous, often crescent-shaped hugging the skull's inner curve.",
        "reliability": "high"
    },
}

RELIABILITY_LABEL = {
    "high": ("#2ecc71", "Well-validated"),
    "moderate": ("#f39c12", "Moderate confidence"),
    "low": ("#e74c3c", "Limited training data"),
}

CLASSIFIER_CLASSES = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subdural"
]

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    detector = YOLO("finalmodel.pt")
    sam = SAM("mobile_sam.pt")
    
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSIFIER_CLASSES))
    # Ensure this file exists in your directory or handle the FileNotFoundError
    classifier.load_state_dict(torch.load("classifier_best.pt", map_location="cpu"))
    classifier.eval()
    
    return detector, sam, classifier

detector, sam_model, classifier = load_models()

# ============================================================
# CLASSIFIER TRANSFORM
# ============================================================

classifier_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🧠 NeuroScan AI</h1>
    <p>Intracranial Hemorrhage Detection · Subtype Classification · Segmentation — Research Prototype</p>
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ Research/educational prototype only. NOT for clinical use — not a diagnostic tool.")

# ============================================================
# UPLOAD + SETTINGS
# ============================================================

col_upload, col_settings = st.columns([2.5, 1])

with col_settings:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    conf_threshold = st.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)
    st.markdown("</div>", unsafe_allow_html=True)

with col_upload:
    uploaded_file = st.file_uploader("Upload a brain-window CT slice (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ============================================================
# INFERENCE
# ============================================================

if uploaded_file:
    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    image_h, image_w = img_np.shape[:2]

    # ========================================================
    # YOLO DETECTION
    # ========================================================
    with st.spinner("Running detection..."):
        det_results = detector.predict(
            source=img_np,
            conf=conf_threshold,
            iou=0.3,
            max_det=5,
            verbose=False
        )

    det_result = det_results[0]
    boxes = det_result.boxes.xyxy.cpu().numpy()
    det_confs = det_result.boxes.conf.cpu().numpy()

    # ========================================================
    # OVERLAY & SAM SEGMENTATION
    # ========================================================
    overlay = img_np.copy()
    detections = []

    if len(boxes) > 0:
        with st.spinner("Segmenting..."):
            sam_results = sam_model(img_np, bboxes=boxes, verbose=False)

        sam_result = sam_results[0]
        
        if sam_result.masks is not None:
            mask_polygons = sam_result.masks.xy
        else:
            mask_polygons = []

        # ====================================================
        # PROCESS EACH DETECTION
        # ====================================================
        for i, (box, det_c) in enumerate(zip(boxes, det_confs)):
            x1, y1, x2, y2 = map(int, box)

            # ------------------------------------------------
            # CLASSIFIER CROP
            # ------------------------------------------------
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * 0.30)
            pad_y = int(bh * 0.30)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(image_w, x2 + pad_x)
            cy2 = min(image_h, y2 + pad_y)

            crop = img_np[cy1:cy2, cx1:cx2]

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------
            if crop.size == 0:
                subtype = "Unknown"
                cls_conf = 0.0
            else:
                crop_r = cv2.resize(crop, (224, 224))
                tensor = classifier_tf(crop_r).unsqueeze(0)
                
                with torch.no_grad():
                    logits = classifier(tensor)
                    probs = torch.softmax(logits, dim=1)
                    idx = probs.argmax(1).item()
                    cls_conf = probs[0, idx].item()
                subtype = CLASSIFIER_CLASSES[idx]

            # =================================================
            # SUBTYPE INFO
            # =================================================
            info = SUBTYPE_INFO.get(
                subtype,
                {"color": "#4f8cff", "icon": "⚪", "blurb": "", "reliability": "moderate"}
            )

            color_hex = info["color"]
            color_rgb = tuple(int(color_hex.lstrip("#")[j:j + 2], 16) for j in (0, 2, 4))

            # =================================================
            # SAM MASK OVERLAY
            # =================================================
            if i < len(mask_polygons):
                polygon = mask_polygons[i]

                if polygon is not None and len(polygon) >= 3:
                    polygon = np.asarray(polygon, dtype=np.int32)
                    mask_canvas = np.zeros((image_h, image_w), dtype=np.uint8)
                    cv2.fillPoly(mask_canvas, [polygon], 1)
                    
                    color_mask = np.zeros_like(img_np)
                    color_mask[mask_canvas.astype(bool)] = color_rgb
                    
                    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.45, 0)

            # =================================================
            # YOLO BOX
            # =================================================
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_rgb, 2)

            detections.append({
                "subtype": subtype,
                "det_conf": float(det_c),
                "cls_conf": float(cls_conf),
                "info": info
            })

    # ========================================================
    # SUMMARY
    # ========================================================
    st.markdown("### Summary")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{len(detections)}</div>
                <div class="stat-label">Regions Flagged</div>
            </div>
        """, unsafe_allow_html=True)

    with s2:
        avg_det = np.mean([d["det_conf"] for d in detections]) if detections else 0
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{avg_det:.0%}</div>
                <div class="stat-label">Avg Detection Conf.</div>
            </div>
        """, unsafe_allow_html=True)

    with s3:
        n_types = len(set(d["subtype"] for d in detections))
        st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{n_types}</div>
                <div class="stat-label">Subtypes Identified</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # ========================================================
    # IMAGE + DETAILS
    # ========================================================
    img_col, detail_col = st.columns([1.3, 1])

    with img_col:
        st.image(
            overlay,
            caption="YOLO Detection + ResNet Classification + MobileSAM Segmentation",
            use_container_width=True
        )

    with detail_col:
        if not detections:
            st.success("✅ No hemorrhage detected above the confidence threshold.")
        else:
            for d in detections:
                info = d["info"]
                rel_color, rel_label = RELIABILITY_LABEL[info["reliability"]]

                st.markdown(f"""
                    <div class="detection-card" style="--accent-color:{info['color']}">
                        <div class="detection-title">{info['icon']} {d['subtype']}</div>
                        <span class="badge" style="background:{rel_color}22; color:{rel_color};">
                            {rel_label}
                        </span>
                        <div class="detection-meta" style="margin-top:8px;">
                            {info['blurb']}
                        </div>
                        <div style="font-size:0.8rem; color:#8b93b0;">Detection confidence</div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{d['det_conf'] * 100}%; background:{info['color']};"></div>
                        </div>
                        <div style="font-size:0.8rem; color:#8b93b0;">Classification confidence</div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{d['cls_conf'] * 100}%; background:{info['color']};"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# ============================================================
# SUBTYPE REFERENCE PANEL
# ============================================================
st.markdown("---")
st.markdown("### Hemorrhage Subtype Reference")

ref_cols = st.columns(4)

for col, (name, info) in zip(ref_cols, SUBTYPE_INFO.items()):
    rel_color, rel_label = RELIABILITY_LABEL[info["reliability"]]
    
    with col:
        st.markdown(f"""
            <div class="info-card" style="border-top:3px solid {info['color']};">
                <div style="font-weight:700; color:#f0f2f8;">
                    {info['icon']} {name}
                </div>
                <span class="badge" style="background:{rel_color}22; color:{rel_color}; margin:6px 0; display:inline-block;">
                    {rel_label}
                </span>
                <div style="font-size:0.82rem; color:#8b93b0; margin-top:6px;">
                    {info['blurb']}
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.caption(
    "Detector: YOLOv11n, single-class hemorrhage presence (Hssayeni CT-ICH dataset). "
    "Classifier: ResNet18, 4 subtypes — Subarachnoid excluded (3 patients, insufficient data). "
    "Segmentation: box-prompted MobileSAM, not fine-tuned. "
    "Patient-level splits used throughout. "
    "Reliability badges reflect training data volume per subtype, not clinical validation."
)