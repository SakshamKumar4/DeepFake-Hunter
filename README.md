# Deepfake Hunter

A machine learning-powered web application that detects whether uploaded face images are **real** or **AI-generated (deepfake)** using a trained **MobileNetV2** model. The app is built with **Streamlit** for an interactive frontend and **TensorFlow** for deep learning inference.

---

## Features

- Upload face images (`jpg`, `jpeg`, `png`, `webp`)
- Detect **Authentic**, **Deepfake**, or **Uncertain**
- Real-time prediction with confidence scores
- Advanced settings for threshold tuning
- Clean futuristic UI
- Public sharing using **ngrok**

---

## How the Project Works

```text
User Uploads Image
        ↓
Image Preprocessing
        ↓
MobileNetV2 Model Prediction
        ↓
Fake Probability Score
        ↓
Final Verdict Displayed
```

### Step-by-Step

1. **Image Upload**  
   User uploads a face image through the Streamlit web interface.

2. **Preprocessing**  
   - resized to `160x160`
   - converted to RGB
   - transformed into NumPy array
   - normalized using MobileNetV2 preprocessing

3. **Model Prediction**  
   The trained `.h5` model predicts:
   - `0.00` = Real
   - `1.00` = Fake

4. **Decision Logic**
   - High fake probability → **Deepfake Detected**
   - Low fake probability → **Authentic Image**
   - Middle range → **Uncertain**

5. **Output**
   - Final verdict
   - Confidence score
   - Fake score
   - Real score

---

## Tech Stack

- Frontend: Streamlit
- Backend: Python
- ML Framework: TensorFlow
- Model: MobileNetV2
- Image Processing: Pillow, OpenCV
- Deployment: ngrok

---

## Project Structure

```text
Deepfake-Hunter/
│── app_enhanced.py
│── final_working_model.h5
│── requirements.txt
│── README.md
```

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/yourusername/deepfake-hunter.git
cd deepfake-hunter
```

### Install Dependencies

```bash
pip install streamlit pyngrok tensorflow pillow numpy opencv-python
```

---

## Run Locally

```bash
streamlit run app_enhanced.py
```

Open:

```text
http://localhost:8501
```

---

## Run Publicly with ngrok

### Install ngrok

```bash
pip install pyngrok
```

### Add Auth Token

```bash
ngrok authtoken YOUR_TOKEN_HERE
```

### Start App

```bash
streamlit run app_enhanced.py --server.port 8501 --server.address 0.0.0.0
```

### Create Tunnel

```python
from pyngrok import ngrok
print(ngrok.connect(8501))
```

---

## Google Colab Commands

```python
!pip install streamlit pyngrok tensorflow pillow numpy opencv-python
!streamlit run app_enhanced.py --server.port 8501 --server.address 0.0.0.0 &
from pyngrok import ngrok
print(ngrok.connect(8501))
```

---

## Example Output

```text
Prediction: DEEPFAKE DETECTED
Confidence: 91%
Fake Score: 0.9123
Real Score: 0.0877
```

---

## Future Improvements

- Video deepfake detection
- Real-time webcam scanning
- Heatmap visualization
- Larger dataset training
- Browser extension

---

## Authors

Saksham, Bhavneet & Sayana  
Bennett University, Greater Noida

---

## License

For academic and educational use only.
