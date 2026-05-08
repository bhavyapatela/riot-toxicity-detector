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
:root {{
  --bg:#061016;
  --panel:#0d1416;
  --muted:#9aa0a6;
  --accent:#ff4757;
}}

[data-testid="stAppViewContainer"] {{
  background:
    linear-gradient(180deg, rgba(6,10,18,0.62), rgba(6,10,18,0.72)),
    {splash_css};
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

.block-container, .stApp {{
  color: #e6eef0 !important;
  background: transparent !important;
  padding-top: 110px !important;
}}

h1 {{
  font-size: 44px !important;
  font-weight: 800 !important;
  color: var(--accent) !important;
  letter-spacing: 0.6px;
}}

.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
  border-radius: 14px !important;
  padding: 18px !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  box-shadow: 0 6px 28px rgba(0,0,0,0.45) !important;
}}

.stButton>button {{
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  border: none !important;
  font-weight: 600 !important;
}}

textarea, .stTextArea>div>div>textarea {{
  background-color: rgba(15,20,20,0.62) !important;
  color: #e6eef0 !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 8px !important;
}}

.small-muted {{
  color: #cbd5e1 !important;
  font-size:13px;
}}

.header-accent {{
  height: 6px;
  width: 180px;
  border-radius: 8px;
  background: linear-gradient(90deg, rgba(255,71,87,0.95), rgba(255,71,87,0.45));
  margin-top: 10px;
}}

.logo-top-right {{
  position: fixed;
  top: 90px;
  right: 10px;
  z-index: 99999;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
}}

.logo-top-right img {{
  width: 80px;
  height: auto;
  border-radius: 10px;
  background: rgba(0,0,0,0.12);
  border: 2px solid rgba(0,0,0,0.2);
  box-shadow: 0 10px 30px rgba(0,0,0,0.45);
  display: block;
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


def render_result_card(label, color, action, suggestion, toxicity_score, proba):
    meter_percent = max(0, min(100, toxicity_score * 10))

    html_card = f"""
    <div style="
        padding:20px;
        border-radius:16px;
        background:#071018;
        border:1px solid {color};
        box-shadow:0 0 24px {color}55;
        max-width:560px;
        font-family:Arial, sans-serif;
    ">

        <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">

            <div style="
                width:130px;
                height:130px;
                border-radius:50%;
                background:conic-gradient({color} {meter_percent}%, rgba(255,255,255,0.14) 0);
                display:flex;
                align-items:center;
                justify-content:center;
            ">

                <div style="
                    width:92px;
                    height:92px;
                    border-radius:50%;
                    background:#071018;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:26px;
                    font-weight:900;
                    color:{color};
                ">
                    {toxicity_score:.1f}
                </div>

            </div>

            <div>
                <h2 style="color:{color};margin:0 0 8px 0;">
                    {label}
                </h2>

                <p style="color:#cbd5e1;margin:0;">
                    Toxicity Score: <b>{toxicity_score:.1f}/10</b>
                </p>

                <p style="color:#cbd5e1;margin:0;">
                    Original Probability: <b>{proba:.3f}</b>
                </p>

                <br>

                <span style="
                    background:{color};
                    color:#071018;
                    padding:7px 12px;
                    border-radius:8px;
                    font-weight:800;
                ">
                    {action}
                </span>

                <p style="color:#cbd5e1;margin-top:14px;">
                    Suggestion: <b>{suggestion}</b>
                </p>
            </div>

        </div>
    </div>
    """

    components.html(html_card, height=220)

# -------------------- logo --------------------
if logo_uri:
    logo_html = f"<div class='logo-top-right'><img src='{logo_uri}' alt='logo' /></div>"
else:
    logo_html = """
    <div class='logo-top-right'>
        <div style='color:#9aa0a6;padding:6px 10px;background:rgba(0,0,0,0.25);border-radius:8px;font-size:12px'>
            Add riot_logo.png to assets/
        </div>
    </div>
    """

st.markdown(logo_html, unsafe_allow_html=True)


# -------------------- header --------------------
st.markdown("<h1>Riot — Toxicity Detector Demo</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='small-muted'>
    Demo: A baseline toxicity detector using TF-IDF + Logistic Regression.
    The model returns a toxicity score from <b>0 to 10</b> and suggests a moderation action.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='header-accent'></div>", unsafe_allow_html=True)
st.markdown("---")


# -------------------- session state --------------------
if "user_text" not in st.session_state:
    st.session_state["user_text"] = "You are trash and I hate you"


def load_sample(text: str):
    st.session_state["user_text"] = text


# -------------------- main layout --------------------
col_main, col_right = st.columns([2, 1])

with col_main:
    st.markdown(
        """
        <div class='card'>
            <h3>Live test your text</h3>
            <p class='small-muted'>
                Type or paste chat messages. The system returns a toxicity score,
                risk level, and moderation suggestion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Enter chat / comment text",
        key="user_text",
        height=160,
    )

    st.markdown("---")

    threshold = st.slider(
        "Temporary ban threshold",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
        step=0.1,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Predict", key="predict_btn"):
            try:
                text_to_score = st.session_state.get("user_text", "")

                probs = predict_proba([text_to_score])
                proba = float(probs[0])
                toxicity_score = proba * 10

                details = get_toxicity_details(toxicity_score)

                if toxicity_score >= threshold:
                    details["action"] = "TEMPORARY BAN"
                    details["suggestion"] = "Score crossed your selected ban threshold"

                render_result_card(
                    details["label"],
                    details["color"],
                    details["action"],
                    details["suggestion"],
                    toxicity_score,
                    proba,
                )

                expl = explain_tokens(text_to_score)

                if expl is not None:
                    top_pos, top_neg = expl

                    with st.expander("Show token contributions"):
                        if top_pos:
                            st.subheader("Top toxic signals")
                            for tok, score in top_pos[:8]:
                                st.write(f"{html.escape(tok)} — {score:.4f}")

                        if top_neg:
                            st.subheader("Top safe signals")
                            for tok, score in top_neg[:8]:
                                st.write(f"{html.escape(tok)} — {score:.4f}")

                else:
                    st.info("Token-level explanations not available for this model.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col_b:
        st.markdown(
            """
            <div class='card'>
                <h4>Quick samples</h4>
                <p class='small-muted'>Click to load a sample into the input box.</p>
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
            "She is a great player",
            "Report this cheater",
        ]

        for s in samples:
            st.button(
                s,
                key=f"sample_{s}",
                on_click=load_sample,
                args=(s,),
            )

        st.write("Current text:")
        st.code(st.session_state.get("user_text", ""))


with col_right:
    st.markdown("<div class='card'><h4>Model Status</h4></div>", unsafe_allow_html=True)

    if tfidf is None or clf is None:
        st.warning("TF-IDF baseline model not found. Train the baseline to use this demo.")
    else:
        st.success("TF-IDF baseline ready")
        st.write("Model: TF-IDF + Logistic Regression")

    if embeddings_clf is not None:
        st.info("Embeddings-based model detected and will be used first.")

    st.markdown("---")
    st.markdown(
        """
        <div class='small-muted'>
            Score guide:
            <br>0–3 → Safe
            <br>4–6 → Warning
            <br>7–8 → Toxic / Mute
            <br>9–10 → Highly Toxic / Temporary Ban
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- batch upload --------------------
st.markdown("---")
st.header("Batch predictions")

uploaded = st.file_uploader("Upload CSV column: comment_text", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)

        if "comment_text" not in df.columns:
            st.error("CSV must contain a column named 'comment_text'")

        else:
            st.info(f"Running predictions on {len(df)} rows...")

            texts = df["comment_text"].astype(str).tolist()
            probs = predict_proba(texts)

            df["toxicity_probability"] = probs
            df["toxicity_score"] = probs * 10

            def batch_action(score):
                details = get_toxicity_details(score)
                if score >= threshold:
                    return "TEMPORARY BAN"
                return details["action"]

            df["action"] = df["toxicity_score"].apply(batch_action)

            st.dataframe(df.head(100))

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")


# -------------------- footer --------------------
st.markdown("---")
st.markdown(
    """
    <div class='small-muted'>
        Built as a demo for a Riot-themed toxicity detector.
        This app uses a TF-IDF + Logistic Regression baseline saved in <code>models/</code>.
        <br><br>
        Limitations: may produce false positives and false negatives.
        Do not use this for automated moderation without human review.
    </div>
    """,
    unsafe_allow_html=True,
)