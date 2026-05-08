# streamlit_toxicity_app.py
# Run:
# python -m streamlit run streamlit_toxicity_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import html
import streamlit.components.v1 as components

# -------------------- helper: data uri for images --------------------
def image_data_uri(p: Path):
    if not p.exists():
        return None

    ext = p.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    try:
        b = p.read_bytes()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


# -------------------- page config --------------------
st.set_page_config(page_title="Riot — Toxicity Detector", layout="wide")


# -------------------- assets --------------------
ASSETS = Path("assets")
logo_png = ASSETS / "riot_logo.png"
logo_jpg = ASSETS / "riot_logo.jpg"
splash_jpg = ASSETS / "splash.jpg"

logo_uri = image_data_uri(logo_png) or image_data_uri(logo_jpg)
splash_uri = image_data_uri(splash_jpg)
splash_css = f'url("{splash_uri}")' if splash_uri else "none"


# -------------------- CSS --------------------
RIOT_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

:root {{
  --riot-red: #ff4655;
  --riot-dark: #0f1923;
  --riot-bg: #060a0e;
  --riot-gray: #ece8e1;
  --glass: rgba(255, 255, 255, 0.03);
  --glass-border: rgba(255, 255, 255, 0.1);
  --neon-glow: 0 0 15px rgba(255, 70, 85, 0.4);
}}

[data-testid="stAppViewContainer"] {{
  background:
    linear-gradient(180deg, rgba(6,10,14,0.85), rgba(6,10,14,0.95)),
    {splash_css};
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  font-family: 'Inter', sans-serif;
}}

.block-container {{
  padding-top: 6rem !important;
  max-width: 1200px !important;
}}

/* Glassmorphism Card */
.glass-card {{
  background: var(--glass) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid var(--glass-border) !important;
  border-radius: 16px !important;
  padding: 24px !important;
  transition: all 0.3s ease;
}}

/* Hero Section */
.hero-container {{
  text-align: center;
  padding: 40px 20px 60px;
  margin-bottom: 20px;
  position: relative;
}}

.hero-title {{
  font-size: 64px !important;
  font-weight: 800 !important;
  color: var(--riot-gray) !important;
  text-transform: uppercase;
  letter-spacing: -1px;
  margin-bottom: 10px !important;
}}

.hero-accent {{
  height: 4px;
  width: 200px;
  background: var(--riot-red);
  margin: 0 auto 30px;
  border-radius: 2px;
  box-shadow: var(--neon-glow);
  animation: accentPulse 2s infinite ease-in-out;
}}

@keyframes accentPulse {{
  0% {{ width: 150px; opacity: 0.6; }}
  50% {{ width: 250px; opacity: 1; }}
  100% {{ width: 150px; opacity: 0.6; }}
}}

/* Metric Grid */
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 40px;
}}

.metric-card-wrapper {{
  background: rgba(15, 25, 35, 0.6);
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
  min-height: 170px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  transition: all 0.3s ease;
}}

.metric-card-wrapper:hover {{
  border-color: var(--riot-red);
  box-shadow: var(--neon-glow);
  transform: translateY(-5px);
}}

.metric-label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: #9aa0a6;
  margin-bottom: 10px;
}}

.metric-value {{
  font-size: 28px;
  font-weight: 800;
  margin: 5px 0;
}}

.metric-badge {{
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 800;
  display: inline-block;
  margin-top: 10px;
}}

/* CTA Buttons */
.cta-button {{
  display: inline-block;
  padding: 12px 30px;
  border-radius: 4px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  text-decoration: none;
  transition: all 0.3s ease;
  min-width: 200px;
  text-align: center;
}}

.cta-primary {{
  background: var(--riot-red);
  color: white !important;
}}

.cta-primary:hover {{
  background: #ff5c69;
  box-shadow: 0 0 20px rgba(255, 70, 85, 0.4);
}}

.cta-secondary {{
  background: transparent;
  color: var(--riot-red) !important;
  border: 1px solid var(--riot-red);
}}

.cta-secondary:hover {{
  background: rgba(255, 70, 85, 0.1);
  box-shadow: 0 0 15px rgba(255, 70, 85, 0.2);
}}

/* Custom scrollbar */
::-webkit-scrollbar {{
  width: 8px;
}}
::-webkit-scrollbar-track {{
  background: var(--riot-bg);
}}
::-webkit-scrollbar-thumb {{
  background: #333;
  border-radius: 10px;
}}
::-webkit-scrollbar-thumb:hover {{
  background: var(--riot-red);
}}
</style>
"""

st.markdown(RIOT_CSS, unsafe_allow_html=True)


# -------------------- model paths --------------------
MODEL_DIR = Path("models")
TFIDF_PATH = MODEL_DIR / "tfidf.joblib"
CLF_PATH = MODEL_DIR / "baseline_lr.joblib"
EMB_MODEL = MODEL_DIR / "embeddings_clf.joblib"


@st.cache_resource
def load_models():
    tf = None
    clf = None
    emb = None

    if TFIDF_PATH.exists():
        try:
            tf = joblib.load(TFIDF_PATH)
        except Exception as e:
            st.warning(f"Failed to load TF-IDF model: {e}")

    if CLF_PATH.exists():
        try:
            clf = joblib.load(CLF_PATH)
        except Exception as e:
            st.warning(f"Failed to load classifier: {e}")

    if EMB_MODEL.exists():
        try:
            emb = joblib.load(EMB_MODEL)
        except Exception:
            emb = None

    return tf, clf, emb


tfidf, clf, embeddings_clf = load_models()


# -------------------- prediction functions --------------------
def predict_proba(texts):
    if embeddings_clf is not None:
        try:
            if (
                isinstance(embeddings_clf, dict)
                and "embedder" in embeddings_clf
                and "clf" in embeddings_clf
            ):
                emb_vecs = embeddings_clf["embedder"].encode(texts)
                probs = embeddings_clf["clf"].predict_proba(emb_vecs)[:, 1]
                return probs
        except Exception:
            pass

    if tfidf is None or clf is None:
        raise RuntimeError("Model artifacts not found. Train baseline first.")

    X = tfidf.transform(texts)
    probs = clf.predict_proba(X)[:, 1]
    return probs


def get_toxicity_details(toxicity_score):
    if toxicity_score <= 3:
        return {
            "label": "SAFE",
            "color": "#22c55e",
            "action": "NO ACTION",
            "suggestion": "No moderation needed",
        }

    elif toxicity_score <= 6:
        return {
            "label": "WARNING",
            "color": "#facc15",
            "action": "WARN USER",
            "suggestion": "Send a warning message to the user",
        }

    elif toxicity_score <= 8:
        return {
            "label": "TOXIC",
            "color": "#fb923c",
            "action": "MUTE",
            "suggestion": "Mute the user temporarily",
        }

    else:
        return {
            "label": "HIGHLY TOXIC",
            "color": "#ef4444",
            "action": "TEMPORARY BAN",
            "suggestion": "Temporarily ban the user",
        }


def explain_tokens(text):
    if tfidf is None or clf is None:
        return None

    try:
        feat_names = tfidf.get_feature_names_out()
        X = tfidf.transform([text])
        coefs = clf.coef_[0]
        xarr = X.toarray()[0]
        contributions = xarr * coefs

        top_pos_idx = np.argsort(-contributions)[:12]
        top_neg_idx = np.argsort(contributions)[:12]

        top_pos = [
            (feat_names[i], float(contributions[i]))
            for i in top_pos_idx
            if xarr[i] > 0
        ]

        top_neg = [
            (feat_names[i], float(contributions[i]))
            for i in top_neg_idx
            if xarr[i] > 0
        ]

        return top_pos, top_neg

    except Exception:
        return None


def render_hero_section():
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">Riot Toxicity Detector</h1>
            <p class="hero-subtitle">AI-powered real-time moderation system for gaming communities</p>
            <div class="hero-accent"></div>
            <div class="glass-card hero-desc-card">
                Detect toxic behavior, assign moderation severity, and recommend actions instantly using NLP and Machine Learning.
            </div>
            <div style="display: flex; gap: 20px; justify-content: center; margin-top: 30px;">
                <a href="#live-detection" class="cta-button cta-primary">Run Detection</a>
                <a href="#batch-predictions" class="cta-button cta-secondary">View Analytics</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_metric_cards(score, label, action, confidence, color):
    st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card-wrapper">
                <div class="metric-label">Toxicity Score</div>
                <div class="metric-value" style="color: {color}; text-shadow: 0 0 10px {color}66;">{score:.1f}</div>
                <div style="font-size: 9px; color: #666; letter-spacing: 1px;">SCALE 0-10</div>
            </div>
            <div class="metric-card-wrapper">
                <div class="metric-label">Risk Level</div>
                <div class="metric-badge" style="background: {color}22; color: {color}; border: 1px solid {color}44;">{label}</div>
                <div style="font-size: 9px; color: #666; margin-top: 8px; letter-spacing: 1px;">SEVERITY</div>
            </div>
            <div class="metric-card-wrapper">
                <div class="metric-label">Rec. Action</div>
                <div class="metric-value" style="font-size: 16px; color: var(--riot-gray); line-height: 1.4;">{action}</div>
                <div style="font-size: 9px; color: #666; letter-spacing: 1px;">PROTOCOL</div>
            </div>
            <div class="metric-card-wrapper">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" style="color: #4ade80;">{confidence:.1f}%</div>
                <div style="width: 80%; height: 3px; background: rgba(255,255,255,0.1); border-radius: 2px; margin: 10px auto 0;">
                    <div style="width: {confidence}%; height: 100%; background: #4ade80; border-radius: 2px; box-shadow: 0 0 8px #4ade8088;"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_result_card(label, color, action, suggestion, toxicity_score, proba):
    meter_percent = max(0, min(100, toxicity_score * 10))

    html_content = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        body {{ margin: 0; padding: 10px; background: transparent; font-family: 'Inter', sans-serif; color: #ece8e1; overflow: hidden; }}
        .result-card {{
            padding:25px;
            border-radius:16px;
            background: rgba(15, 25, 35, 0.9);
            border:1px solid {color};
            box-shadow:0 0 30px {color}22;
            display: flex;
            align-items: center;
            gap: 30px;
        }}
        .meter-container {{
            width:120px;
            height:120px;
            border-radius:50%;
            background:conic-gradient({color} {meter_percent}%, rgba(255,255,255,0.05) 0);
            display:flex;
            align-items:center;
            justify-content:center;
            position: relative;
        }}
        .meter-inner {{
            width:90px;
            height:90px;
            border-radius:50%;
            background:#0f1923;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:24px;
            font-weight:900;
            color:{color};
            box-shadow: inset 0 0 15px rgba(0,0,0,0.5);
        }}
        .content {{ flex: 1; }}
        .label {{ color:{color}; margin:0 0 5px 0; text-transform: uppercase; letter-spacing: 2px; font-weight: 800; }}
        .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }}
        .stat-label {{ font-size: 10px; color: #9aa0a6; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-val {{ font-weight: 600; font-size: 15px; }}
        .action-box {{ background:{color}15; color:{color}; padding:12px 18px; border-radius:6px; border: 1px solid {color}33; font-weight:800; font-size: 13px; letter-spacing: 1px; }}
    </style>
    <div class="result-card">
        <div class="meter-container">
            <div class="meter-inner">{toxicity_score:.1f}</div>
        </div>
        <div class="content">
            <div class="label">{label} DETECTED</div>
            <div class="stats">
                <div>
                    <div class="stat-label">Model Confidence</div>
                    <div class="stat-val">{proba:.4f}</div>
                </div>
                <div>
                    <div class="stat-label">Severity Level</div>
                    <div class="stat-val">{toxicity_score:.1f} / 10</div>
                </div>
            </div>
            <div class="action-box">PROTOCOL: {action}</div>
            <p style="color:#9aa0a6; margin:15px 0 0; font-size: 13px;">
                <span style="color: #666; text-transform: uppercase; font-size: 10px; font-weight: 700;">System Recommendation:</span><br>
                {suggestion}
            </p>
        </div>
    </div>
    """
    components.html(html_content, height=260)

# -------------------- navbar / logo --------------------
# Logo removed as per user request


# -------------------- header --------------------
# render_hero_section() called in main flow below


# -------------------- session state --------------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "You are trash and I hate you"

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

def load_sample(text: str):
    st.session_state["user_text"] = text

# -------------------- main layout --------------------
# 1. Hero Section
render_hero_section()
st.markdown('<div id="live-detection"></div>', unsafe_allow_html=True)

# 2. Dashboard Metrics (Row of 4)
if st.session_state["last_result"]:
    res = st.session_state["last_result"]
    render_metric_cards(
        res["score"],
        res["label"],
        res["action"],
        res["proba"] * 100,
        res["color"]
    )
else:
    # Placeholder metrics
    render_metric_cards(0.0, "WAITING", "N/A", 0.0, "#9aa0a6")

st.write("")
st.write("")

# 3. Analysis Section
col_main, col_right = st.columns([2, 1])

with col_main:
    st.markdown(
        """
        <div class='glass-card'>
            <h3 style="margin-top:0; color: var(--riot-gray);">Analysis Console</h3>
            <p style="color: #9aa0a6; font-size: 14px;">
                Enter player communication logs below for real-time toxicity scoring and moderation recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.write("")

    user_text = st.text_area(
        "Enter chat / comment text",
        key="user_text",
        height=140,
        label_visibility="collapsed",
        placeholder="Type player message here..."
    )

    col_btn, col_sld = st.columns([1, 2])
    with col_btn:
        predict_clicked = st.button("Run Detection", key="predict_btn")
    
    with col_sld:
        threshold = st.slider(
            "Auto-Ban Threshold",
            min_value=0.0,
            max_value=10.0,
            value=8.5,
            step=0.1,
            help="Scores above this will trigger a Temporary Ban."
        )

    if predict_clicked:
        try:
            text_to_score = st.session_state.get("user_text", "")
            if not text_to_score.strip():
                st.warning("Please enter some text first.")
            else:
                probs = predict_proba([text_to_score])
                proba = float(probs[0])
                toxicity_score = proba * 10
                details = get_toxicity_details(toxicity_score)

                if toxicity_score >= threshold:
                    details["action"] = "TEMPORARY BAN"
                    details["suggestion"] = "Score crossed your selected ban threshold"

                # Update session state for metrics persistence
                st.session_state["last_result"] = {
                    "score": toxicity_score,
                    "label": details["label"],
                    "action": details["action"],
                    "proba": proba,
                    "color": details["color"],
                    "suggestion": details["suggestion"]
                }
                st.rerun()

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Detailed Results (if result exists)
    if st.session_state["last_result"]:
        res = st.session_state["last_result"]
        st.write("---")
        render_result_card(
            res["label"],
            res["color"],
            res["action"],
            res["suggestion"],
            res["score"],
            res["proba"],
        )

        # Token Explanations
        expl = explain_tokens(st.session_state["user_text"])
        if expl is not None:
            top_pos, top_neg = expl
            with st.expander("🔍 View Token Contribution Analysis"):
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    if top_pos:
                        st.markdown("<h4 style='color: #ef4444;'>Toxic Signals</h4>", unsafe_allow_html=True)
                        for tok, score in top_pos[:8]:
                            st.write(f"**{html.escape(tok)}** `{score:.4f}`")
                with c2:
                    if top_neg:
                        st.markdown("<h4 style='color: #22c55e;'>Safe Signals</h4>", unsafe_allow_html=True)
                        for tok, score in top_neg[:8]:
                            st.write(f"**{html.escape(tok)}** `{score:.4f}`")
                st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # Samples Section
    st.markdown(
        """
        <div class='glass-card'>
            <h4 style="margin-top:0;">Quick Samples</h4>
            <p style="color: #9aa0a6; font-size: 13px;">Load test cases instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    samples = [
        "You are an idiot",
        "Great shot! nice game",
        "I will find you and kill you",
        "Get rekt noob",
        "I love this community",
        "Report this cheater",
    ]

    for s in samples:
        st.button(s, key=f"sample_{s}", on_click=load_sample, args=(s,))

    st.write("")
    
    # Model Status
    st.markdown("<div class='glass-card'><h4>System Status</h4>", unsafe_allow_html=True)
    if tfidf is None or clf is None:
        st.error("⚠️ Backend Offline")
    else:
        st.success("✅ Model Active")
        st.markdown("<div style='font-size:12px; color:#9aa0a6;'>Engine: TF-IDF + Logistic Regression</div>", unsafe_allow_html=True)
    
    if embeddings_clf is not None:
        st.info("🚀 Neural Embeddings Active")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- batch predictions --------------------
st.markdown('<div id="batch-predictions"></div>', unsafe_allow_html=True)
st.write("")
st.write("---")
st.header("Batch Predictions")

with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (required column: 'comment_text')", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if "comment_text" not in df.columns:
                st.error("CSV must contain a column named 'comment_text'")
            else:
                st.info(f"Processing {len(df)} entries...")
                texts = df["comment_text"].astype(str).tolist()
                probs = predict_proba(texts)
                df["toxicity_probability"] = probs
                df["toxicity_score"] = probs * 10

                def batch_action(score):
                    details = get_toxicity_details(score)
                    if score >= threshold: return "TEMPORARY BAN"
                    return details["action"]

                df["action"] = df["toxicity_score"].apply(batch_action)
                st.dataframe(df.head(100), use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Moderation Report",
                    data=csv,
                    file_name="toxicity_report.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Batch processing failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- footer --------------------
st.write("")
st.write("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px; padding: 40px;'>
        Riot Toxicity Detector &copy; 2026 | Powered by AI/NLP<br>
        This is a demonstration system. Automated actions should be verified by human moderators.
    </div>
    """,
    unsafe_allow_html=True,
)
