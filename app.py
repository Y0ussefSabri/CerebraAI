import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO, SAM
from PIL import Image

st.set_page_config(page_title="CerebraAI", page_icon="🧠", layout="wide")

# ============================================================
# STYLING — warm orange/amber gradient theme, dark base
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #120b08; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 1100px; }

    .main-header {
        background: linear-gradient(135deg, #ff9d5c 0%, #d9531e 35%, #3a1c10 80%, #120b08 100%);
        padding: 1.6rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
        border: 1px solid #4a2a18; position: relative; overflow: hidden;
    }
    .main-header::before {
        content: ""; position: absolute; top: -40%; right: -10%;
        width: 60%; height: 180%;
        background: radial-gradient(circle, rgba(255,180,120,0.35) 0%, rgba(255,180,120,0) 70%);
        pointer-events: none;
    }
    .main-header h1 {
        font-size: 2rem; font-weight: 800; margin: 0; color: #fff5ec;
        letter-spacing: -0.02em; position: relative;
    }
    .main-header p { color: #ffe0c2; margin: 4px 0 12px 0; font-size: 0.88rem; position: relative; }
    .tag-pill {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 700; margin-right: 8px; position: relative;
    }
    .tag-dark { background: #1c1310; color: #ffcda0; border: 1px solid #4a2a18; }
    .tag-orange { background: #e8631f; color: #fff5ec; }

    section[data-testid="stSidebar"] { background-color: #17100c; border-right: 1px solid #3a2418; }
    section[data-testid="stSidebar"] * { color: #f0dcc9 !important; }

    .info-card {
        background: #1c130e; border: 1px solid #3a2418; border-radius: 12px;
        padding: 0.7rem 0.9rem; margin-bottom: 0.6rem;
    }
    .detection-card {
        background: linear-gradient(135deg, #1c130e 0%, #1c130e 60%, #2a1810 100%);
        border-radius: 12px; padding: 0.85rem 1.05rem;
        margin-bottom: 0.6rem; border-left: 4px solid var(--accent-color);
    }
    .detection-title { font-size: 1rem; font-weight: 700; color: #fff5ec; margin-bottom: 0.15rem; }
    .detection-meta { color: #c9a98c; font-size: 0.78rem; margin: 0.35rem 0 0.45rem 0; line-height: 1.35; }
    .conf-row { display: flex; align-items: center; gap: 8px; font-size: 0.72rem; color: #c9a98c; margin-top: 4px; }
    .conf-bar-bg { background: #3a2418; border-radius: 5px; height: 6px; flex: 1; }
    .conf-bar-fill { height: 6px; border-radius: 5px; background: linear-gradient(90deg, #ff9d5c, #e8631f); }

    .stat-row { display: flex; gap: 0.7rem; margin-bottom: 0.9rem; }
    .stat-box {
        background: #1c130e; border: 1px solid #3a2418; border-radius: 12px;
        padding: 0.7rem; text-align: center; flex: 1;
    }
    .stat-number {
        font-size: 1.5rem; font-weight: 800; line-height: 1.1;
        background: linear-gradient(90deg, #ff9d5c, #e8631f);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-label { color: #a3876f; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

    .badge {
        display: inline-block; padding: 2px 9px; border-radius: 20px;
        font-size: 0.68rem; font-weight: 700;
    }
    .method-note {
        background: #17100c; border: 1px solid #3a2418; border-radius: 8px;
        padding: 0.55rem 0.75rem; font-size: 0.72rem; color: #a3876f; margin-top: 0.5rem; line-height: 1.4;
    }
    h3 { font-size: 1.05rem !important; margin-bottom: 0.6rem !important; color: #fff5ec; font-weight: 700 !important; }
    .ref-mini {
        background: #1c130e; border: 1px solid #3a2418; border-radius: 10px;
        padding: 0.6rem 0.7rem; font-size: 0.74rem; margin-bottom: 0.4rem;
    }
    .ref-mini b { color: #fff5ec; }

    /* Streamlit widget accents */
    .stSlider [data-baseweb="slider"] div[role="slider"] { background-color: #e8631f !important; }
    .stCheckbox [data-testid="stMarkdownContainer"] { color: #f0dcc9 !important; }
    div[data-testid="stFileUploader"] section { background-color: #1c130e; border: 1px dashed #4a2a18; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SUBTYPE METADATA — warm palette per subtype
# ============================================================
SUBTYPE_INFO = {
    'Epidural': {'color': '#ff5c5c', 'icon': '🔴',
        'blurb': 'Bleeding between skull and dura, often arterial. Biconvex (lens-shaped) on CT.', 'reliability': 'high'},
    'Intraparenchymal': {'color': '#ff9d5c', 'icon': '🟠',
        'blurb': 'Bleeding within brain tissue, often hypertension/trauma-linked.', 'reliability': 'moderate'},
    'Intraventricular': {'color': '#ffd23f', 'icon': '🟡',
        'blurb': "Bleeding into the ventricular system; hydrocephalus risk.", 'reliability': 'low'},
    'Subdural': {'color': '#e8631f', 'icon': '🟤',
        'blurb': "Bleeding between dura and arachnoid, typically venous, crescent-shaped.", 'reliability': 'high'},
}
RELIABILITY_LABEL = {
    'high': ('#2ecc71', 'Well-validated'),
    'moderate': ('#f39c12', 'Moderate confidence'),
    'low': ('#e74c3c', 'Limited data'),
}
CLASSIFIER_CLASSES = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subdural']
MIN_MASK_AREA_FRAC = 0.001

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    detector = YOLO("finalmodel.pt")
    sam = SAM("mobile_sam.pt")
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSIFIER_CLASSES))
    classifier.load_state_dict(torch.load("classifier_best.pt", map_location='cpu'))
    classifier.eval()
    return detector, sam, classifier

detector, sam_model, classifier = load_models()
classifier_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧠 CerebraAI</h1>
    <p>Intracranial Hemorrhage Detection · Classification · Segmentation — Research Prototype</p>
    <span class="tag-pill tag-dark">YOLOv11 · ResNet18 · SAM</span>
    <span class="tag-pill tag-orange">Computer Vision · 2026</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — controls
# ============================================================
with st.sidebar:
    st.markdown("### Settings")
    uploaded_file = st.file_uploader("Upload CT slice", type=["jpg", "jpeg", "png"])
    conf_threshold = st.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)
    mask_opacity = st.slider("Mask opacity", 0.2, 0.9, 0.55, 0.05)
    show_box = st.checkbox("Show raw detection box", value=False,
                            help="The box is a coarse localization step; the segmented mask is the precise, validated output.")
    st.markdown("---")
    st.markdown(
        '<div class="method-note">⚠️ Research/educational prototype only. NOT for clinical use.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="method-note">Detector: YOLOv11n (Hssayeni CT-ICH). Classifier: ResNet18, 4 subtypes '
        '(Subarachnoid excluded — insufficient data). Segmentation: box-prompted MobileSAM.</div>',
        unsafe_allow_html=True
    )

# ============================================================
# INFERENCE + RESULTS
# ============================================================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    tmp_path = "temp_upload.jpg"
    cv2.imwrite(tmp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    with st.spinner("Analyzing..."):
        det_results = detector.predict(source=tmp_path, conf=conf_threshold, iou=0.3,
                                        max_det=5, verbose=False, augment=True)
        boxes = det_results[0].boxes.xyxy.cpu().numpy()
        det_confs = det_results[0].boxes.conf.cpu().numpy()

        overlay = img_np.copy()
        detections = []
        img_area = img_np.shape[0] * img_np.shape[1]

        if len(boxes) > 0:
            sam_results = sam_model(tmp_path, bboxes=boxes)
            masks = sam_results[0].masks.data.cpu().numpy()

            for (x1, y1, x2, y2), det_c, mask in zip(boxes, det_confs, masks):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                mask_resized = cv2.resize(
                    mask.astype(np.uint8), (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)

                if mask_resized.sum() < MIN_MASK_AREA_FRAC * img_area:
                    continue

                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(img_np.shape[1], x2 + pad_x), min(img_np.shape[0], y2 + pad_y)
                crop = img_np[cy1:cy2, cx1:cx2]

                if crop.size == 0:
                    subtype, cls_conf = "Unknown", 0.0
                else:
                    crop_r = cv2.resize(crop, (224, 224))
                    tensor = classifier_tf(crop_r).unsqueeze(0)
                    with torch.no_grad():
                        probs = torch.softmax(classifier(tensor), dim=1)
                        idx = probs.argmax(1).item()
                        cls_conf = probs[0, idx].item()
                    subtype = CLASSIFIER_CLASSES[idx]

                info = SUBTYPE_INFO.get(subtype, {'color': '#e8631f', 'icon': '⚪', 'blurb': '', 'reliability': 'moderate'})
                color_rgb = tuple(int(info['color'].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

                # --- solid fill ---
                color_layer = np.zeros_like(img_np)
                color_layer[mask_resized.astype(bool)] = color_rgb
                overlay = cv2.addWeighted(overlay, 1.0, color_layer, mask_opacity, 0)

                # --- contour outline so the region reads clearly even at low opacity ---
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color_rgb, 2)

                # --- label anchored to the mask's own extent ---
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    mx, my, mw, mh = cv2.boundingRect(largest)
                    label = f"{subtype} {det_c:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ly1 = max(0, my - th - 6)
                    cv2.rectangle(overlay, (mx, ly1), (mx + tw + 6, my), color_rgb, -1)
                    cv2.putText(overlay, label, (mx + 3, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 15), 1)

                # --- optional raw box ---
                if show_box:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_rgb, 1)

                detections.append({'subtype': subtype, 'det_conf': det_c, 'cls_conf': cls_conf, 'info': info})

    # ---- Compact stat row ----
    avg_det = np.mean([d['det_conf'] for d in detections]) if detections else 0
    n_types = len(set(d['subtype'] for d in detections))
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-number">{len(detections)}</div><div class="stat-label">Flagged</div></div>
        <div class="stat-box"><div class="stat-number">{avg_det:.0%}</div><div class="stat-label">Avg Conf.</div></div>
        <div class="stat-box"><div class="stat-number">{n_types}</div><div class="stat-label">Subtypes</div></div>
    </div>
    """, unsafe_allow_html=True)

    img_col, detail_col = st.columns([1.4, 1])
    with img_col:
        st.image(overlay, use_container_width=True)
    with detail_col:
        if not detections:
            st.success("✅ No hemorrhage detected above threshold.")
        else:
            for d in detections:
                info = d['info']
                rel_color, rel_label = RELIABILITY_LABEL[info['reliability']]
                st.markdown(f"""
                <div class="detection-card" style="--accent-color:{info['color']}">
                    <div class="detection-title">{info['icon']} {d['subtype']}</div>
                    <span class="badge" style="background:{rel_color}22; color:{rel_color};">{rel_label}</span>
                    <div class="detection-meta">{info['blurb']}</div>
                    <div class="conf-row">Det <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{d['det_conf']*100}%;"></div></div> {d['det_conf']:.0%}</div>
                    <div class="conf-row">Cls <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{d['cls_conf']*100}%;"></div></div> {d['cls_conf']:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("👈 Upload a CT slice in the sidebar to begin.")

    st.markdown("### Subtype Reference")
    ref_cols = st.columns(4)
    for col, (name, info) in zip(ref_cols, SUBTYPE_INFO.items()):
        rel_color, rel_label = RELIABILITY_LABEL[info['reliability']]
        with col:
            st.markdown(f"""
            <div class="ref-mini" style="border-top: 2px solid {info['color']};">
                <b>{info['icon']} {name}</b><br>
                <span class="badge" style="background:{rel_color}22; color:{rel_color};">{rel_label}</span>
                <div style="color:#a3876f; margin-top:4px;">{info['blurb']}</div>
            </div>
            """, unsafe_allow_html=True)