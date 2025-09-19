import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import io
import pandas as pd

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

# --- Functions ---
@st.cache_resource
def load_model():
    try:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(1280, len(LABEL_NAMES))
        model_path = os.path.join(os.path.dirname(__file__), 'best_multi_breed_efficientnetb0_6class.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
