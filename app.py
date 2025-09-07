import streamlit as st

# --- Custom CSS to fix gap ---
st.markdown("""
    <style>
        /* Remove default Streamlit top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Center everything */
        .main {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        /* Reduce spacing between widgets */
        div[data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("### 🐂🐃 Smart Cattle & Buffalo Breed Classifier")
st.caption("Upload an image of cattle, buffalo, sheep, or similar animals to get breed prediction")

# --- Upload Box ---
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# --- Button ---
if st.button("🔍 Analyze Image"):
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Image uploaded successfully (Model execution will be added later).")
    else:
        st.warning("⚠️ Please upload an image first.")
