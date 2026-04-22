import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import os
import tempfile
from collections import deque

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Hunter · Deepfake Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ── GLOBAL CSS (PROFESSIONAL STYLING) ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Courier+Prime:wght@400;700&display=swap');

/* ── BASE STYLES ── */
* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    font-family: 'Poppins', sans-serif !important;
    color: #ffffff !important;
}

/* Main container */
.main { 
    background: transparent !important; 
}

[data-testid="stMainBlockContainer"] {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

.block-container {
    max-width: 900px !important;
    padding: 30px 20px !important;
    margin: 0 auto !important;
}

/* ── HIDE STREAMLIT ELEMENTS ── */
#MainMenu, footer, header { display: none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 10px;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: rgba(100,200,255,0.1) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(100,200,255,0.2) !important;
    color: #90aeff !important;
    padding: 12px 24px !important;
}

[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(100,200,255,0.25) !important;
    border-color: rgba(100,200,255,0.5) !important;
    color: #64c8ff !important;
}

/* ── HERO SECTION ── */
.hero-container {
    text-align: center;
    padding: 40px 0 30px 0;
    background: rgba(255,255,255,0.03);
    border-radius: 20px;
    margin-bottom: 30px;
    border: 1px solid rgba(100,200,255,0.2);
}

.hero-badge {
    display: inline-block;
    background: rgba(100,200,255,0.15);
    border: 1px solid rgba(100,200,255,0.4);
    color: #64c8ff;
    padding: 8px 16px;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 15px;
    text-transform: uppercase;
    font-family: 'Courier Prime', monospace;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #64c8ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1rem;
    color: #b0c4ff;
    font-weight: 300;
    margin-bottom: 5px;
    line-height: 1.6;
}

/* ── FILE UPLOADER STYLING ── */
[data-testid="stFileUploader"] {
    background: rgba(100,200,255,0.08) !important;
    border: 2px dashed rgba(100,200,255,0.3) !important;
    border-radius: 15px !important;
    padding: 30px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(100,200,255,0.6) !important;
    background: rgba(100,200,255,0.12) !important;
}

[data-testid="stFileUploader"] label {
    color: #90aeff !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}

/* ── COLUMNS & LAYOUT ── */
[data-testid="stHorizontalBlock"] {
    gap: 20px !important;
}

[data-testid="stColumn"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    border: 1px solid rgba(100,200,255,0.15) !important;
}

/* ── RESULT CARD ── */
.result-container {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 25px;
    margin-top: 20px;
    border: 1px solid rgba(100,200,255,0.2);
    text-align: center;
    line-height: 2;
}

.result-fake {
    border-color: rgba(255,100,100,0.4) !important;
    background: rgba(255,100,100,0.08) !important;
}

.result-real {
    border-color: rgba(100,255,150,0.4) !important;
    background: rgba(100,255,150,0.08) !important;
}

.result-uncertain {
    border-color: rgba(255,180,80,0.4) !important;
    background: rgba(255,180,80,0.08) !important;
}

.verdict-text {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 15px 0;
    letter-spacing: 0.5px;
}

.verdict-fake .verdict-text { color: #ff6464; }
.verdict-real .verdict-text { color: #64ff96; }
.verdict-uncertain .verdict-text { color: #ffa850; }

/* ── SCORE BOXES ── */
.score-container {
    display: flex;
    gap: 15px;
    margin: 20px 0;
    justify-content: center;
}

.score-box {
    background: rgba(100,200,255,0.1);
    border: 1px solid rgba(100,200,255,0.3);
    border-radius: 12px;
    padding: 15px 20px;
    flex: 1;
    text-align: center;
    min-width: 140px;
}

.score-label {
    font-size: 0.75rem;
    color: #7fa3d1;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
    font-family: 'Courier Prime', monospace;
}

.score-value {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'Courier Prime', monospace;
}

.score-fake { color: #ff8080; }
.score-real { color: #80ffb0; }

/* ── INFO BOX ── */
.info-box {
    background: rgba(100,200,255,0.08);
    border-left: 3px solid rgba(100,200,255,0.5);
    padding: 15px;
    border-radius: 8px;
    margin: 20px 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c0d4ff;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(100,200,255,0.2) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] > button {
    background: transparent !important;
    color: #90aeff !important;
    font-weight: 600 !important;
}

/* ── METRIC ── */
[data-testid="stMetricDelta"] {
    color: #64c8ff !important;
}

/* ── TEXT ELEMENTS ── */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 600 !important;
}

p, span, label {
    color: #d0dcff !important;
    font-weight: 400 !important;
}

/* ── FOOTER ── */
.footer-text {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(100,200,255,0.2);
    color: #7fa3d1;
    font-size: 0.9rem;
    line-height: 1.8;
}

.footer-text a {
    color: #64c8ff;
    text-decoration: none;
    font-weight: 600;
}

/* ── DIVIDER ── */
hr {
    border: none;
    border-top: 1px solid rgba(100,200,255,0.15);
    margin: 30px 0;
}

/* ── SMOOTHING ── */
* {
    transition: all 0.3s ease;
}

/* ── RESPONSIVE ── */
@media (max-width: 768px) {
    .hero-title { font-size: 2.5rem; }
    .score-container { flex-direction: column; }
    [data-testid="stColumn"] { padding: 15px; }
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    """Load the pre-trained deepfake detection model"""
    return tf.keras.models.load_model("final_working_model.h5")

# Loading state
loading_placeholder = st.empty()
with loading_placeholder.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align:center; padding:20px;"><div style="font-size:1.2rem; color:#64c8ff;">⏳ Loading AI Model...</div></div>', unsafe_allow_html=True)

model = load_deepfake_model()
loading_placeholder.empty()

# ── HERO SECTION ──────────────────────────────────────────────────────────────
st.markdown("""<div class="hero-container">
    <div class="hero-badge">🔍 Advanced AI Analysis</div>
    <div class="hero-title">DeepFake Hunter</div>
    <div class="hero-subtitle">
        Professional Deepfake & AI-Generated Face Detection<br>
        <span style="font-size: 0.9rem; color: #a0c4ff;">Upload images or videos to detect AI-manipulations instantly</span>
    </div>
</div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── PREDICTION FUNCTION ───────────────────────────────────────────────────────
def predict_deepfake(image: Image.Image, invert_labels: bool, threshold: float):
    """
    Predict whether an image is fake or real
    
    Args:
        image: PIL Image object
        invert_labels: Whether to invert the prediction
        threshold: Confidence threshold (0.4-0.6)
    
    Returns:
        Tuple of (verdict, fake_prob, real_prob)
    """
    # Preprocess image
    img_resized = image.resize((160, 160)).convert("RGB")
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction
    model_output = float(model.predict(img_array, verbose=0)[0][0])

    # Apply inversion if needed
    if invert_labels:
        fake_probability = 1 - model_output
    else:
        fake_probability = model_output
    
    real_probability = 1 - fake_probability

    # Determine verdict based on threshold
    upper_bound = 0.5 + (threshold / 2)
    lower_bound = 0.5 - (threshold / 2)

    if fake_probability > upper_bound:
        verdict = "🚨 DEEPFAKE DETECTED"
    elif fake_probability < lower_bound:
        verdict = "✅ AUTHENTIC IMAGE"
    else:
        verdict = "⚠️ UNCERTAIN RESULT"

    return verdict, fake_probability, real_probability

def extract_frames_from_video(video_path: str, sample_interval: int = 5):
    """Extract frames from video at specified interval"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames, total_frames, fps, duration

def predict_video(video_path: str, invert_labels: bool, threshold: float, sample_interval: int = 5):
    """
    Predict deepfake status for video by analyzing sampled frames
    
    Returns:
        Tuple of (verdict, avg_fake_prob, frame_predictions, total_frames, fps, duration)
    """
    frames, total_frames, fps, duration = extract_frames_from_video(video_path, sample_interval)
    
    if not frames:
        return None, None, None, 0, 0, 0
    
    frame_predictions = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, frame in enumerate(frames):
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # Preprocess
        img_resized = pil_image.resize((160, 160)).convert("RGB")
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        model_output = float(model.predict(img_array, verbose=0)[0][0])
        
        if invert_labels:
            fake_probability = 1 - model_output
        else:
            fake_probability = model_output
        
        frame_predictions.append(fake_probability)
        
        # Update progress
        progress = (idx + 1) / len(frames)
        progress_bar.progress(progress)
        status_text.text(f"🎬 Processing: {idx + 1}/{len(frames)} frames")
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate average
    avg_fake_prob = np.mean(frame_predictions)
    real_prob = 1 - avg_fake_prob
    
    # Determine verdict based on threshold
    upper_bound = 0.5 + (threshold / 2)
    lower_bound = 0.5 - (threshold / 2)
    
    if avg_fake_prob > upper_bound:
        verdict = "🚨 DEEPFAKE DETECTED"
    elif avg_fake_prob < lower_bound:
        verdict = "✅ AUTHENTIC VIDEO"
    else:
        verdict = "⚠️ UNCERTAIN RESULT"
    
    return verdict, avg_fake_prob, frame_predictions, total_frames, fps, duration

def display_results(verdict, fake_prob, real_prob, confidence_threshold, is_video=False):
    """Display prediction results"""
    fake_pct = int(fake_prob * 100)
    real_pct = int(real_prob * 100)
    
    st.markdown("---")
    
    # Verdict
    if 'DEEPFAKE' in verdict:
        st.error("🚨 **DEEPFAKE DETECTED**", icon=None)
        st.metric("Detection Result", "FAKE")
    elif 'AUTHENTIC' in verdict:
        st.success("✅ **AUTHENTIC**", icon=None)
        st.metric("Detection Result", "REAL")
    else:
        st.warning("⚠️ **UNCERTAIN RESULT**", icon=None)
        st.metric("Detection Result", "UNCERTAIN")
    
    # Metrics
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.metric("🔴 Fake Score", f"{fake_prob:.4f}")
        st.metric("Fake %", f"{fake_pct}%")
    
    with metric_col2:
        st.metric("🟢 Real Score", f"{real_prob:.4f}")
        st.metric("Real %", f"{real_pct}%")
    
    st.markdown("---")
    
    # Confidence bar
    st.write(f"**Confidence Level:** {fake_pct}%")
    st.progress(fake_prob)
    
    # Info section
    media_type = "VIDEO" if is_video else "IMAGE"
    st.info(f"""
    **📊 Analysis Results:**
    
    • Model Output: {fake_prob:.6f}
    • Classification: {'DEEPFAKE' if 'DEEPFAKE' in verdict else 'AUTHENTIC' if 'AUTHENTIC' in verdict else 'UNCERTAIN'}
    • Confidence: ±{int(confidence_threshold*100)}%
    • Media Type: {media_type}
    """)

# ── SETTINGS SECTION ──────────────────────────────────────────────────────────
st.markdown("### ⚙️ Settings & Configuration")
settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    flip_prediction = st.checkbox(
        "🔄 Invert Labels",
        value=False,
        help="Enable this if real images are being marked as fake"
    )

with settings_col2:
    sensitivity_option = st.selectbox(
        "🎯 Confidence Level",
        options=["Low (0.6)", "Medium (0.5)", "High (0.4)"],
        index=1
    )
    confidence_threshold = float(sensitivity_option.split("(")[1].split(")")[0])

st.markdown("<hr>", unsafe_allow_html=True)

# ── TABS FOR IMAGE AND VIDEO ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸 Image Detection", "🎬 Video Detection"])

# ── IMAGE TAB ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📤 Upload Your Image")
    uploaded_image = st.file_uploader(
        "📁 Select an image file (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # Load and process image
        image = Image.open(uploaded_image)
        image = ImageOps.exif_transpose(image)
        
        # Show image and results in columns
        img_col, result_col = st.columns([1, 1], gap="large")
        
        with img_col:
            st.markdown('<div class="score-label">📸 UPLOADED IMAGE</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True, output_format="auto")
            st.markdown(f'<div class="score-label" style="text-align:center; margin-top:10px;">Size: {image.size[0]}×{image.size[1]} px</div>', unsafe_allow_html=True)
        
        with result_col:
            # Predict
            with st.spinner("🔍 Analyzing image..."):
                time.sleep(0.3)
                verdict, fake_prob, real_prob = predict_deepfake(
                    image, flip_prediction, confidence_threshold
                )
            
            display_results(verdict, fake_prob, real_prob, confidence_threshold, is_video=False)

# ── VIDEO TAB ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📤 Upload Your Video")
    
    video_col1, video_col2 = st.columns(2)
    
    with video_col1:
        sample_rate = st.selectbox(
            "🎯 Sampling Rate",
            options=[1, 3, 5, 10],
            index=2,
            help="Process every Nth frame (higher = faster but less accurate)"
        )
    
    with video_col2:
        max_duration = st.number_input(
            "⏱️ Max Video Duration (seconds)",
            value=60,
            min_value=10,
            max_value=300,
            step=10
        )
    
    uploaded_video = st.file_uploader(
        "📁 Select a video file (MP4, MOV, AVI)",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_path = tmp_file.name
        
        try:
            # Check video duration
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Display video info
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("📹 Duration", f"{duration:.2f}s")
            with info_col2:
                st.metric("🎬 Total Frames", f"{total_frames}")
            with info_col3:
                st.metric("⚡ FPS", f"{fps:.1f}")
            
            # Check duration limit
            if duration > max_duration:
                st.error(f"❌ Video exceeds maximum duration of {max_duration}s (current: {duration:.2f}s)")
            else:
                # Show video preview
                st.video(uploaded_video)
                
                if st.button("🔍 Analyze Video", use_container_width=True):
                    st.markdown("---")
                    st.markdown("### 📊 Processing Video...")
                    
                    with st.spinner("Extracting and analyzing frames..."):
                        verdict, avg_fake_prob, frame_predictions, total_frames, fps, duration = predict_video(
                            tmp_path, flip_prediction, confidence_threshold, sample_rate
                        )
                    
                    if verdict is not None:
                        # Display results
                        display_results(verdict, avg_fake_prob, 1 - avg_fake_prob, confidence_threshold, is_video=True)
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        
                        # Detailed analysis
                        with st.expander("📊 **Detailed Frame Analysis**"):
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            
                            with detail_col1:
                                st.metric("Frames Analyzed", len(frame_predictions))
                                st.metric("Sample Interval", f"Every {sample_rate} frame(s)")
                            
                            with detail_col2:
                                st.metric("Avg Fake Score", f"{avg_fake_prob:.4f}")
                                st.metric("Min Score", f"{min(frame_predictions):.4f}")
                            
                            with detail_col3:
                                st.metric("Max Score", f"{max(frame_predictions):.4f}")
                                st.metric("Std Dev", f"{np.std(frame_predictions):.4f}")
                            
                            # Frame predictions chart
                            st.markdown("### Frame-by-Frame Predictions")
                            chart_data = {
                                "Frame Index": list(range(len(frame_predictions))),
                                "Fake Probability": frame_predictions
                            }
                            st.line_chart({"Fake Probability": frame_predictions})
                            
                            st.info("""
                            **📈 Interpretation:**
                            - **Flat line near 0**: Consistent REAL video
                            - **Flat line near 1**: Consistent FAKE video
                            - **Fluctuating line**: Mix of real and AI-generated content
                            - **Sharp spikes**: Potential deepfake sections
                            """)
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**DeepFake Hunter** - Professional Deepfake Detection System (Image + Video)  
Built with TensorFlow & MobileNetV2 | B.Tech Capstone Project  
BENNETT University, Greater Noida, India  

📧 Developer: [Saksham](mailto:sakshamkumar4494@gmail.com)
""")
