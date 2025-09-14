# Smart Cattle & Buffalo Breed Classifier

A machine learning application for classifying cattle and buffalo breeds using deep learning. This project uses an EfficientNet-B0 model trained on images of six different breeds to provide accurate breed predictions through a user-friendly Streamlit web interface.

## 🐄 Supported Breeds

- Ayrshire
- Brown Swiss
- Guernsey
- Hariana
- Holstein Friesian
- Jersey

## ✨ Features

- **Web-based Interface**: Easy-to-use Streamlit application for image upload and prediction
- **Deep Learning Model**: EfficientNet-B0 architecture for high accuracy classification
- **Real-time Prediction**: Instant breed classification from uploaded images
- **Responsive Design**: Modern UI with custom styling and responsive layout
- **Image Preprocessing**: Automatic image resizing, normalization, and augmentation

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd SIH
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at ```https://ourfirstprototype.streamlit.app/```.

### How to Use

1. **Upload Image**: Click on the upload area and select a JPG, JPEG, or PNG image of cattle/buffalo
2. **Analyze**: Click the "🔍 Analyze Image" button to get the breed prediction
3. **View Results**: The predicted breed will be displayed alongside the uploaded image

## 📁 Project Structure

```
SIH/
├── app.py                 # Streamlit web application
├── model.py               # Model training script
├── requirements.txt       # Python dependencies
├── best_multi_breed_efficientnetb0_6class.pth  # Trained model weights
└── README.md             # Project documentation
```

## 🧠 Model Details

### Architecture
- **Base Model**: EfficientNet-B0
- **Input Size**: 224x224 pixels
- **Output Classes**: 6 breeds
- **Framework**: PyTorch

### Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Epochs**: 10 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss with class weights
- **Data Augmentation**: Random horizontal flip, rotation, color jitter

### Preprocessing
- Resize to 256x256, center crop to 224x224
- Normalization with ImageNet mean and std
- Conversion to PyTorch tensors

## 📊 Training Script

The `model.py` file contains the complete training pipeline including:

- Custom dataset class for loading images
- Data augmentation and preprocessing
- Model training with validation
- Metrics calculation (accuracy, precision, recall, F1-score)
- Confusion matrix generation
- Early stopping and model checkpointing
- Results visualization and export

### Running Training

To train the model (requires dataset):
```bash
python model.py
```

**Note**: Training requires access to the original dataset directories specified in the script.

## 🔧 Dependencies

- `streamlit` - Web application framework
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision library for PyTorch
- `pillow` - Image processing library

## 📈 Performance

The model achieves high accuracy on the validation set with comprehensive evaluation metrics including per-class precision, recall, and F1-scores.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open-source. Please check the license file for more details.

## 📞 Support

For questions or issues, please open an issue in the repository or contact the maintainers.

---

**Built with ❤️ using PyTorch and Streamlit**
