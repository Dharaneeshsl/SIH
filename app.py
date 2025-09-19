import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os

# Initialize session state for caching
if 'model' not in st.session_state:
    try:
        # Load the model
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(1280, 6)  # 6 classes
        model_path = os.path.join(os.path.dirname(__file__), 'best_multi_breed_efficientnetb0_6class.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        st.session_state.model = model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Breed labels
LABEL_NAMES = ["Ayrshire", "Brown Swiss", "Guernsey", "Hariana", "Holstein Friesian", "Jersey"]

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        /* Page Layout */
        .block-container {
            padding: 2rem 1rem;
            max-width: 900px;
            margin: auto;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        /* App Title */
        .app-title {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            font-size: clamp(1.5rem, 5vw, 2rem);
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }
        .app-caption {
            text-align: center;
            font-size: clamp(0.8rem, 2.5vw, 0.9rem);
            color: #666;
            margin-bottom: 1.5rem;
        }

        /* Upload Box */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #4CAF50;
            border-radius: 12px;
            padding: 1.5rem;
            background-color: #f9f9f9;
            margin-bottom: 1rem;
        }

        /* Buttons */
        div[data-testid="stButton"] > button {
            width: 100%;
            padding: 0.8rem;
            border-radius: 10px;
            font-weight: 600;
            background: linear-gradient(to right, #4CAF50, #2e7d32);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(to right, #66bb6a, #388e3c);
            transform: translateY(-2px);
        }

        /* Uploaded Image */
        .uploaded-image {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        /* Result Box */
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            background: #f0fdf4;
            border: 1px solid #a7f3d0;
            text-align: center;
            font-size: clamp(1.1rem, 3vw, 1.3rem);
            font-weight: 600;
            color: #065f46;
            margin-top: 1rem;
        }

        /* Responsive adjustments */
        @media (max-width: 600px) {
            .block-container {
                padding: 1rem;
            }
            .app-title {
                font-size: 1.5rem;
            }
            .result-box {
                font-size: 1.1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='app-title'>🐂 🐃 Smart Cattle & Buffalo Breed Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='app-caption'>Upload an image (JPG, JPEG, or PNG) to identify the breed of cattle or buffalo</div>", unsafe_allow_html=True)

# --- Upload Box ---
uploaded_file = st.file_uploader("📤 Upload an image file", type=["jpg", "jpeg", "png"], help="Upload a clear image of a cattle or buffalo for breed prediction")

# --- Prediction & Display ---
if st.button("🔍 Analyze Image"):
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing image..."):
                # Open and preprocess the image
                image = Image.open(uploaded_file).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)

                # Run inference
                with torch.no_grad():
                    output = st.session_state.model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    confidence, predicted = torch.max(probabilities, 0)
                    predicted_class = predicted.item()

                # Get predicted breed and confidence
                predicted_breed = LABEL_NAMES[predicted_class] if 0 <= predicted_class < len(LABEL_NAMES) else "Unknown Breed"
                confidence_score = confidence.item() * 100

                # Display results
                col1, col2 = st.columns([1.2, 1])
                with col1:
                    st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True, output_format="JPEG", clamp=True)
                with col2:
                    st.markdown(
                        f"<div class='result-box'>"
                        f"✅ Predicted Breed: {predicted_breed}<br>"
                        f"Confidence: {confidence_score:.2f}%"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.warning("⚠️ Please upload an image first.")
