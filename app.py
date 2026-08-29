"""
Date Fruit Analyzer
--------------------
Detects date fruit clusters in an image with a custom-trained YOLOv8 model,
then classifies each detected fruit by variety and maturity stage with two
further YOLOv8 classification heads.

Originally built as a Flask app for a PFE (graduation project) on date-fruit
detection for automated harvesting; rebuilt here as a single-file Streamlit app.
"""

import io
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
HISTORY_DB = BASE_DIR / "history.db"
HISTORY_IMAGES_DIR = BASE_DIR / "history_outputs"
HISTORY_IMAGES_DIR.mkdir(exist_ok=True)

DETECTION_MODEL_PATH = MODELS_DIR / "Detection.pt"
VARIETY_MODEL_PATH = MODELS_DIR / "Variety.pt"
MATURITY_MODEL_PATH = MODELS_DIR / "Maturity.pt"

# Class labels the classification heads were trained on.
VARIETY_NAMES = {0: "Boufagous", 1: "Bouisthami", 2: "Boumajhoul", 3: "Kholt"}
MATURITY_NAMES = {0: "Khalal", 1: "Kimri", 2: "Rutab", 3: "Tamar"}

# Colors used for maturity-stage badges, tuned to the actual fruit colors at
# each stage rather than arbitrary category colors.
MATURITY_COLORS = {
    "Kimri": "#5C7A3B",    # unripe green
    "Khalal": "#C98A3E",   # golden yellow
    "Rutab": "#8C4A2F",    # ripening brown
    "Tamar": "#3B2417",    # fully ripe, near-black brown
}

BOX_COLOR = (214, 163, 68)  # gold accent, matches --gold in the CSS below


# --------------------------------------------------------------------------
# Styling — orchard / harvest theme
# --------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #1F2E1A 0%, #24331D 100%);
            color: #F3EAD3;
        }

        h1, h2, h3, .hero-title {
            font-family: 'Fraunces', serif;
            color: #F3EAD3;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0;
            line-height: 1.1;
        }

        .hero-subtitle {
            color: #C9BFA0;
            font-size: 1.05rem;
            margin-top: 0.3rem;
            margin-bottom: 1.4rem;
        }

        .panel {
            background: #2E4526;
            border: 1px solid #46603A;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }

        .ripeness-scale {
            display: flex;
            border-radius: 6px;
            overflow: hidden;
            height: 10px;
            margin: 0.6rem 0 0.2rem 0;
        }
        .ripeness-scale div { flex: 1; }

        .ripeness-labels {
            display: flex;
            justify-content: space-between;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: #C9BFA0;
            letter-spacing: 0.02em;
        }

        .badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: #1F2E1A;
            font-weight: 500;
        }

        .stButton>button {
            background: #D6A344;
            color: #1F2E1A;
            border: none;
            font-weight: 600;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background: #E8B65C;
            color: #1F2E1A;
        }

        [data-testid="stMetricValue"] {
            color: #D6A344;
            font-family: 'Fraunces', serif;
        }

        .footer-note {
            color: #8E9782;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ripeness_scale_html():
    stages = ["Kimri", "Khalal", "Rutab", "Tamar"]
    bars = "".join(
        f'<div style="background:{MATURITY_COLORS[s]};"></div>' for s in stages
    )
    labels = "".join(f"<span>{s}</span>" for s in stages)
    return f"""
    <div class="ripeness-scale">{bars}</div>
    <div class="ripeness-labels">{labels}</div>
    """


# --------------------------------------------------------------------------
# Model loading (cached so weights load once per session)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    detector = YOLO(str(DETECTION_MODEL_PATH))
    variety_clf = YOLO(str(VARIETY_MODEL_PATH))
    maturity_clf = YOLO(str(MATURITY_MODEL_PATH))
    return detector, variety_clf, maturity_clf


# --------------------------------------------------------------------------
# History storage (lightweight SQLite, replaces the original Flask/SQLAlchemy DB)
# --------------------------------------------------------------------------

def init_history_db():
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            num_detected INTEGER,
            results_json TEXT,
            image_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_to_history(filename, results, annotated_image: Image.Image):
    image_id = uuid.uuid4().hex[:12]
    image_path = HISTORY_IMAGES_DIR / f"{image_id}.jpg"
    annotated_image.convert("RGB").save(image_path, quality=88)

    conn = sqlite3.connect(HISTORY_DB)
    conn.execute(
        "INSERT INTO analyses (filename, timestamp, num_detected, results_json, image_path) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(results),
            json.dumps(results),
            str(image_path),
        ),
    )
    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect(HISTORY_DB)
    rows = conn.execute(
        "SELECT id, filename, timestamp, num_detected, results_json, image_path "
        "FROM analyses ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows

def delete_history_entry(entry_id, image_path):
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("DELETE FROM analyses WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()

    img_path = Path(image_path)
    if img_path.exists():
        img_path.unlink()

# --------------------------------------------------------------------------
# Inference pipeline
# --------------------------------------------------------------------------

def get_font(size=16):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def run_pipeline(image: Image.Image, detector, variety_clf, maturity_clf):
    """Detect date clusters, then classify each one by variety and maturity.

    Returns the annotated image and a list of per-detection result dicts.
    """
    image = image.convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(16)

    det_results = detector(image, verbose=False)
    boxes = det_results[0].boxes.xyxy.cpu().numpy() if len(det_results) else []
    det_confs = det_results[0].boxes.conf.cpu().numpy() if len(det_results) else []

    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image.crop((x1, y1, x2, y2))
        if crop.width == 0 or crop.height == 0:
            continue

        variety_pred = variety_clf(crop, verbose=False)[0]
        maturity_pred = maturity_clf(crop, verbose=False)[0]

        variety_label = VARIETY_NAMES.get(variety_pred.probs.top1, "Unknown")
        variety_conf = float(variety_pred.probs.top1conf)
        maturity_label = MATURITY_NAMES.get(maturity_pred.probs.top1, "Unknown")
        maturity_conf = float(maturity_pred.probs.top1conf)

        results.append(
            {
                "id": i + 1,
                "detection_conf": float(det_confs[i]) if len(det_confs) > i else None,
                "variety": variety_label,
                "variety_conf": round(variety_conf, 3),
                "maturity": maturity_label,
                "maturity_conf": round(maturity_conf, 3),
                "box": [x1, y1, x2, y2],
            }
        )

        stage_color = MATURITY_COLORS.get(maturity_label, "#D6A344")
        box_width = max(3, int(min(image.width, image.height) * 0.01))
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=box_width)
        label = f"#{i + 1}  {variety_label} / {maturity_label}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        pad = 3
        draw.rectangle(
            [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad],
            fill=stage_color,
        )
        draw.text((x1, y1), label, fill="#F3EAD3", font=font)

    return annotated, results


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def render_header():
    st.markdown('<div class="hero-title">🌴 Date Fruit Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Detects date clusters, then classifies each fruit '
        "by variety and ripeness stage — built on a custom YOLOv8 pipeline trained on "
        "9,000+ orchard images from Drâa-Tafilalet, Morocco.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel">' + ripeness_scale_html() + "</div>", unsafe_allow_html=True)


def render_analyze_tab(detector, variety_clf, maturity_clf):
    uploaded = st.file_uploader(
        "Upload a photo of a date cluster on the palm", type=["jpg", "jpeg", "png"]
    )

    image = None
    filename = None
    if uploaded:
        image = Image.open(uploaded)
        filename = uploaded.name

    if image is None:
        st.info(
            "Upload a JPG or PNG showing a cluster of dates still on the branch — "
            "the model was trained on in-orchard photos, so shots of loose or "
            "bowled dates won't detect well."
        )
        return

    with st.spinner("Running detection and classification..."):
        annotated, results = run_pipeline(image, detector, variety_clf, maturity_clf)

    if not results:
        st.warning("No date fruit clusters were detected in this image.")
        st.image(image, use_container_width=True)
        return

    st.image(annotated, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Fruits detected", len(results))
    varieties_found = ", ".join(sorted({r["variety"] for r in results}))
    col2.metric("Varieties found", varieties_found)

    df = pd.DataFrame(
        [
            {
                "#": r["id"],
                "Variety": r["variety"],
                "Variety confidence": r["variety_conf"],
                "Maturity": r["maturity"],
                "Maturity confidence": r["maturity_conf"],
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    save_to_history(filename, results, annotated)


def render_history_tab():
    rows = load_history()
    if not rows:
        st.info("No analyses yet — results will appear here after you run one.")
        return

    for entry_id, filename, timestamp, num_detected, results_json, image_path in rows:
        results = json.loads(results_json)
        with st.expander(f"{filename}  ·  {timestamp}  ·  {num_detected} fruit(s)"):
            img_path = Path(image_path)
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            if results:
                df = pd.DataFrame(
                    [
                        {
                            "#": r["id"],
                            "Variety": r["variety"],
                            "Maturity": r["maturity"],
                        }
                        for r in results
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
            if st.button("Delete this entry", key=f"delete_{entry_id}"):
                delete_history_entry(entry_id, image_path)
                st.rerun()


def main():
    st.set_page_config(
        page_title="Date Fruit Analyzer",
        page_icon="🌴",
        layout="centered",
    )
    inject_css()
    init_history_db()

    detector, variety_clf, maturity_clf = load_models()

    render_header()

    tab_analyze, tab_history = st.tabs(["Analyze", "History"])
    with tab_analyze:
        render_analyze_tab(detector, variety_clf, maturity_clf)
    with tab_history:
        render_history_tab()

    st.markdown(
        '<div class="footer-note">Graduation project by Boudaoud Aissa & Bouhani ELMustafa · '
        "École Supérieure de Technologie de Khénifra · 2023–2024</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
