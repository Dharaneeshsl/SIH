```python
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os
import pandas as pd
import numpy as np

# --- Constants ---
LABEL_NAMES = ["Ayrshire", "Brown Swiss", "Guernsey", "Hariana", "Holstein Friesian", "Jersey"]

BREED_DESCRIPTIONS = {
    "Ayrshire": "Ayrshire cattle are known for their efficiency in converting grass to milk, hardiness, and longevity. They originated in Scotland and are typically red and white in color.",
    "Brown Swiss": "Brown Swiss is one of the oldest dairy breeds, originating from Switzerland. They are known for their high milk protein content, heat tolerance, and gentle temperament.",
    "Guernsey": "Guernsey cattle produce high-quality milk with a golden color due to beta-carotene. They are efficient grazers and adaptable to various climates, originating from the Isle of Guernsey.",
    "Hariana": "Hariana is an Indian breed known for its draft power and milk production in tropical conditions. They are hardy, disease-resistant, and well-adapted to hot climates.",
    "Holstein Friesian": "Holstein Friesian is the world's highest-producing dairy breed, known for large volumes of milk. They are typically black and white and originated in the Netherlands.",
    "Jersey": "Jersey cattle produce milk high in butterfat and protein. They are small-sized, efficient converters of feed to milk, and originated from the Isle of Jersey."
}

CONFIDENCE_THRESHOLD = 0.5  # 50% threshold for reliable prediction
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# --- Functions ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(1280, len(LABEL_NAMES))
        model_path = os.path.join(os.path.dirname(__file__), 'best_multi_breed_efficientnetb0_6class.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def predict_breed(image: Image.Image, model):
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        return probabilities
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        :root {
            --primary-color: #4CAF50;
            --secondary-color: #2e7d32;
            --bg-color: #f9f9f9;
            --text-color: #333;
            --accent-color: #065f46;
        }
        [data-theme="dark"] {
            --primary-color: #66bb6a;
            --secondary-color: #388e3c;
            --bg-color: #1e1e1e;
            --text-color: #f0f0f0;
            --accent-color: #a7f3d0;
        }
        .block-container {
            padding: 2rem 1rem;
            max-width: 950px;
            margin: auto;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .app-title {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            font-size: clamp(1.6rem, 5vw, 2.2rem);
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }
        .app-caption {
            text-align: center;
            font-size: clamp(0.85rem, 2.5vw, 0.95rem);
            color: #666;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stFileUploader"] {
            border: 2px dashed var(--primary-color);
            border-radius: 12px;
            padding: 1.5rem;
            background-color: var(--bg-color);
            margin-bottom: 1rem;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            padding: 0.8rem;
            border-radius: 10px;
            font-weight: 600;
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: linear-gradient(to right, #757575, #616161);
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: linear-gradient(to right, #9e9e9e, #757575);
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            background: var(--bg-color);
            border: 1px solid var(--primary-color);
            text-align: center;
            font-size: clamp(1.1rem, 3vw, 1.3rem);
            font-weight: 600;
            color: var(--accent-color);
            margin-top: 1rem;
        }
        .description-box {
            padding: 1rem;
            border-radius: 8px;
            background: #f0fdf4;
            border: 1px solid #a7f3d0;
            font-size: 0.95rem;
            color: #065f46;
            margin-top: 1rem;
        }
        [data-theme="dark"] .description-box {
            background: #2e7d32;
            border: 1px solid #66bb6a;
            color: #f0f0f0;
        }
        .preview-image {
            max-height: 300px;
            object-fit: contain;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        @media (max-width: 600px) {
            .block-container {
                padding: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- Theme Switcher ---
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
theme = st.selectbox("Choose Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
st.session_state.theme = theme
st.markdown(f'<style>[data-theme] {{ --theme: "{theme.lower()}" }}</style>', unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='app-title'>🐂 🐃 Smart Cattle & Buffalo Breed Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='app-caption'>Upload a clear image (JPG, JPEG, or PNG, max 5MB) for AI-powered breed identification</div>", unsafe_allow_html=True)

# --- Upload Box ---
uploaded_file = st.file_uploader(
    "📤 Upload an image file",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a clear, well-lit photo focused on the animal.",
    key="file_uploader"
)

# --- Image Preview ---
if uploaded_file:
    st.image(uploaded_file, caption="Preview", use_column_width=False, width=300, clamp=True, output_format="JPEG")
    st.markdown("<small>Ensure the image clearly shows the animal for accurate results.</small>", unsafe_allow_html=True)

# --- Buttons ---
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    analyze_clicked = st.button("🔍 Analyze Image", type="primary")
with col_btn2:
    reset_clicked = st.button("🔄 Reset", type="secondary")

# --- Reset Logic ---
if reset_clicked:
    st.session_state.pop("file_uploader", None)
    st.rerun()

# --- Prediction & Display ---
if analyze_clicked:
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image first.")
    elif uploaded_file.size > MAX_FILE_SIZE:
        st.warning("⚠️ File size exceeds 5MB limit. Please upload a smaller image.")
    else:
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()
                image = Image.open(uploaded_file).convert('RGB')
                probabilities = predict_breed(image, model)
                if probabilities is None:
                    st.stop()

                top_prob, top_idx = torch.max(probabilities, 0)
                predicted_breed = LABEL_NAMES[top_idx.item()]
                confidence = top_prob.item()

                # Top 3 predictions
                top3 = torch.topk(probabilities, 3)
                top3_breeds = [LABEL_NAMES[i] for i in top3.indices]
                top3_probs = top3.values.numpy() * 100

                # Display results
                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True, output_format="JPEG")

                with col2:
                    if confidence >= CONFIDENCE_THRESHOLD:
                        st.markdown(
                            f"<div class='result-box'>"
                            f"✅ Predicted Breed: {predicted_breed}<br>"
                            f"Confidence: {confidence * 100:.2f}%"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='result-box' style='border-color: #ffcc00; color: #b37400;'>"
                            f"⚠️ Low Confidence Prediction: {predicted_breed}<br>"
                            f"Confidence: {confidence * 100:.2f}% - Try a clearer image"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Breed Description
                st.markdown(
                    f"<div class='description-box'>{BREED_DESCRIPTIONS.get(predicted_breed, 'No description available.')}</div>",
                    unsafe_allow_html=True
                )

                # Probability Chart
                st.subheader("Prediction Breakdown")
                ```chartjs
                {
                    "type": "bar",
                    "data": {
                        "labels": ["Ayrshire", "Brown Swiss", "Guernsey", "Hariana", "Holstein Friesian", "Jersey"],
                        "datasets": [{
                            "label": "Probability (%)",
                            "data": [probabilities[0].item() * 100, probabilities[1].item() * 100, probabilities[2].item() * 100, probabilities[3].item() * 100, probabilities[4].item() * 100, probabilities[5].item() * 100],
                            "backgroundColor": ["#4CAF50", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9", "#E8F5E9"],
                            "borderColor": ["#2E7D32", "#388E3C", "#43A047", "#4CAF50", "#66BB6A", "#81C784"],
                            "borderWidth": 1
                        }]
                    },
                    "options": {
                        "indexAxis": "y",
                        "scales": {
                            "x": {
                                "beginAtZero": true,
                                "max": 100,
                                "title": {
                                    "display": true,
                                    "text": "Probability (%)"
                                }
                            }
                        },
                        "plugins": {
                            "legend": {
                                "display": false
                            },
                            "title": {
                                "display": true,
                                "text": "Breed Prediction Probabilities"
                            }
                        }
                    }
                }
                ```

                # Top 3 Predictions Table
                top3_df = pd.DataFrame({
                    'Breed': top3_breeds,
                    'Probability (%)': [f"{p:.2f}" for p in top3_probs]
                })
                st.table(top3_df)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "Powered by EfficientNet | Developed with Streamlit & PyTorch | <a href='https://x.ai' target='_blank'>xAI</a>"
    "</p>",
    unsafe_allow_html=True
)
```
