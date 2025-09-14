import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Load the model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(1280, 6)  # Assuming 6 classes
model.load_state_dict(torch.load('SIH/best_multi_breed_efficientnetb0_6class.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        /* Page Layout */
        .block-container {
            padding-top: 2rem !important;  /* Fix title cut */
            padding-bottom: 1rem;
            max-width: 800px;
            margin: auto;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        /* App Title */
        .app-title {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 2.5rem; /* Prevent emoji cut-off */
            margin-bottom: 0.5rem;
        }
        .app-caption {
            text-align: center;
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 1.2rem;
        }

        /* Upload Box */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #4CAF50;
            border-radius: 15px;
            padding: 1rem;
            background-color: #f9f9f9;
        }

        /* Buttons */
        div[data-testid="stButton"] > button {
            width: 100%;
            padding: 0.8rem;
            border-radius: 10px;
            font-weight: bold;
            background: linear-gradient(to right, #4CAF50, #2e7d32);
            color: white;
            border: none;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(to right, #66bb6a, #388e3c);
        }

        /* Uploaded Image */
        .uploaded-image {
            width: 100%;
            border-radius: 12px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }

        /* Result Box */
        .result-box {
            padding: 1rem;
            border-radius: 12px;
            background: #f0fdf4;
            border: 1px solid #a7f3d0;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 600;
            color: #065f46;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='app-title'>🐂 🐃 Smart Cattle & Buffalo Breed Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='app-caption'>Upload an image of cattle, buffalo, sheep, or similar animals to get breed prediction</div>", unsafe_allow_html=True)

# --- Upload Box ---
uploaded_file = st.file_uploader("📤 Upload an image file", type=["jpg", "jpeg", "png"])

# --- Prediction & Display ---
if st.button("🔍 Analyze Image"):
    if uploaded_file is not None:
        # Open and preprocess the image
        image = Image.open(uploaded_file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()

        # Labels
        label_names = ["ayrshire", "brown_swiss", "guernsey", "hariana", "holstein_friesian", "jersey"]
        predicted_breed = label_names[predicted_class] if 0 <= predicted_class < len(label_names) else "Unknown Breed"

        # Two columns → image left, result right
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True, output_format="JPEG")
        with col2:
            st.markdown(f"<div class='result-box'>✅ Predicted Breed:<br>{predicted_breed}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload an image first.")
