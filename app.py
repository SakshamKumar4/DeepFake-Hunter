import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Hunter · Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050a0f !important;
    color: #e8edf2 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #0d2137 0%, #050a0f 60%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }

.block-container { 
    max-width: 1100px !important; 
    padding: 2rem 2rem 4rem !important; 
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.3);
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    background: rgba(0,212,255,0.05);
}
.hero-title {
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6b8090;
    font-weight: 400;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(0,212,255,0.25) !important;
    border-radius: 16px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.6) !important;
    background: rgba(0,212,255,0.03) !important;
}
[data-testid="stFileUploader"] label {
    color: #8aaabb !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #4a6572 !important;
}

/* ── Image Card ── */
.img-card {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
}
.img-label {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.15em;
    color: #4a6572;
    text-transform: uppercase;
    padding: 14px 16px 0;
}

/* ── Result Card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
}
.result-fake { 
    background: rgba(255,45,55,0.06); 
    border: 1px solid rgba(255,45,55,0.2);
}
.result-real { 
    background: rgba(0,230,120,0.05); 
    border: 1px solid rgba(0,230,120,0.15);
}
.result-uncertain { 
    background: rgba(255,170,0,0.05); 
    border: 1px solid rgba(255,170,0,0.2);
}

.verdict {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.verdict-fake { color: #ff4757; }
.verdict-real { color: #2ed573; }
.verdict-uncertain { color: #ffa502; }

.verdict-icon { font-size: 2.2rem; margin-bottom: 0.3rem; }

/* ── Confidence Bar ── */
.conf-wrap { margin-top: 0.5rem; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #6b8090;
    margin-bottom: 6px;
}
.bar-track {
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.bar-fill-fake {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff4757, #ff6b81);
    transition: width 0.8s cubic-bezier(.16,1,.3,1);
}
.bar-fill-real {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #2ed573, #7bed9f);
    transition: width 0.8s cubic-bezier(.16,1,.3,1);
}
.bar-fill-uncertain {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ffa502, #ffd32a);
}

/* ── Score Row ── */
.score-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.2rem;
}
.score-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.06);
}
.score-box-label {
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    color: #4a6572;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.score-box-val {
    font-size: 1.2rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.score-fake { color: #ff6b81; }
.score-real { color: #7bed9f; }

/* ── Info Note ── */
.info-note {
    font-size: 0.78rem;
    color: #4a6572;
    border-left: 2px solid rgba(0,212,255,0.3);
    padding-left: 10px;
    line-height: 1.5;
    margin-top: 0.5rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 2rem 0;
}

/* ── Sidebar toggle (settings) ── */
.settings-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
}
.settings-title {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.15em;
    color: #4a6572;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding-top: 2rem;
    color: #2a3a45;
    font-size: 0.78rem;
}
.footer a { color: #00d4ff; text-decoration: none; }

/* ── Streamlit image fix ── */
[data-testid="stImage"] img {
    border-radius: 12px;
}

/* ── Override default text ── */
h1,h2,h3,h4,h5,h6,p,span,label {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Selectbox / toggle styling */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8edf2 !important;
}
[data-testid="stCheckbox"] label {
    color: #8aaabb !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("final_working_model.h5")

with st.spinner("Loading DeepScan AI model..."):
    model = load_model()


# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔍 AI-Powered Analysis · MobileNetV2</div>
    <div class="hero-title">DeepFake Hunter</div>
    <div class="hero-sub">
        Detect AI-generated &amp; manipulated faces with deep learning.<br>
        Upload any face image to instantly verify its authenticity.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── SETTINGS ──────────────────────────────────────────────────────────────────
with st.expander("⚙️ Advanced Settings", expanded=False):
    st.markdown('<div class="settings-title">Model Configuration</div>', unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        flip_labels = st.checkbox(
            "Flip prediction labels",
            value=False,
            help="If real images are wrongly shown as fake, enable this to flip the model's output interpretation."
        )
    with col_s2:
        sensitivity = st.select_slider(
            "Detection sensitivity",
            options=["Low (0.6)", "Medium (0.5)", "High (0.4)"],
            value="Medium (0.5)"
        )
    
    threshold_map = {"Low (0.6)": 0.6, "Medium (0.5)": 0.5, "High (0.4)": 0.4}
    THRESHOLD = threshold_map[sensitivity]
    
    st.markdown(
        f'<div class="info-note">Current threshold: <b>{THRESHOLD}</b> · '
        f'Label flip: <b>{"ON" if flip_labels else "OFF"}</b><br>'
        f'If results seem inverted, enable "Flip prediction labels" above.</div>',
        unsafe_allow_html=True
    )


# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload Face Image")
uploaded_file = st.file_uploader(
    "Drop a face image here or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)


# ── PREDICTION LOGIC ──────────────────────────────────────────────────────────
def predict(img: Image.Image, flip: bool, threshold: float):
    """Returns (verdict, fake_prob, real_prob, css_class, icon, bar_class)"""
    img_resized = img.resize((160, 160)).convert("RGB")
    arr = np.array(img_resized, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    raw = float(model.predict(arr, verbose=0)[0][0])

    # Apply label flip if needed
    fake_prob = (1 - raw) if flip else raw
    real_prob = 1 - fake_prob

    upper = 0.5 + threshold / 2   # e.g. 0.75 at medium
    lower = 0.5 - threshold / 2   # e.g. 0.25 at medium

    if fake_prob > upper:
        verdict, css, icon, bar = "DEEPFAKE DETECTED", "result-fake", "🚨", "bar-fill-fake"
    elif fake_prob < lower:
        verdict, css, icon, bar = "AUTHENTIC IMAGE", "result-real", "✅", "bar-fill-real"
    else:
        verdict, css, icon, bar = "UNCERTAIN", "result-uncertain", "⚠️", "bar-fill-uncertain"

    return verdict, fake_prob, real_prob, css, icon, bar


# ── RENDER RESULT ─────────────────────────────────────────────────────────────
if uploaded_file:
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="img-label">Input Image</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            time.sleep(0.4)  # slight pause for UX
            verdict, fake_prob, real_prob, css_class, icon, bar_class = predict(
                img, flip_labels, THRESHOLD
            )

        fake_pct = int(fake_prob * 100)
        real_pct = int(real_prob * 100)

        if "FAKE" in verdict:
            verdict_cls = "verdict-fake"
        elif "AUTHENTIC" in verdict:
            verdict_cls = "verdict-real"
        else:
            verdict_cls = "verdict-uncertain"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div>
                <div class="verdict-icon">{icon}</div>
                <div class="verdict {verdict_cls}">{verdict}</div>
            </div>
            
            <div class="conf-wrap">
                <div class="conf-label">
                    <span>FAKE PROBABILITY</span>
                    <span>{fake_pct}%</span>
                </div>
                <div class="bar-track">
                    <div class="{bar_class}" style="width:{fake_pct}%"></div>
                </div>
            </div>

            <div class="score-row">
                <div class="score-box">
                    <div class="score-box-label">🔴 Fake Score</div>
                    <div class="score-box-val score-fake">{fake_prob:.4f}</div>
                </div>
                <div class="score-box">
                    <div class="score-box-label">🟢 Real Score</div>
                    <div class="score-box-val score-real">{real_prob:.4f}</div>
                </div>
            </div>

            <div class="info-note">
                Score range: 0.00 (Real) → 1.00 (Fake)<br>
                Confidence threshold: ±{int(THRESHOLD*100)}% from center
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── DETAILS EXPANDER ──
    with st.expander("📊 Technical Details"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Model Output", f"{fake_prob:.6f}" if not flip_labels else f"{1-fake_prob:.6f}")
        c2.metric("Image Resolution", f"{img.size[0]}×{img.size[1]}")
        c3.metric("Model Input", "160×160 px")

        st.markdown(f"""
        <div class="info-note" style="margin-top:1rem">
            <b>How it works:</b> This model uses MobileNetV2 pretrained on ImageNet, 
            fine-tuned on deepfake face datasets. The sigmoid output represents the 
            probability of the image being AI-generated or manipulated.<br><br>
            <b>If results seem wrong:</b> Enable "Flip prediction labels" in Advanced Settings above. 
            Different training datasets use different label conventions (0=Real/1=Fake or 0=Fake/1=Real).
        </div>
        """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class="divider">
<div class="footer">
    Built with ❤️ by <b>Saksham ,Bhavneet & Sayana </b> · Bennett University, Greater Noida<br>
    B.TECH Capstone Project · MobileNetV2 Deep Learning<br>
    <a href="mailto:sakshamkumar4494@gmail.com">SakshamKumar4494@gmail.com</a>
</div>
""", unsafe_allow_html=True)
