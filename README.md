# 🫁 Pneumonia Detection System

A comprehensive deep learning-based system for detecting pneumonia from chest X-ray images using YOLOv8 classification model and a user-friendly Streamlit web application.

## 📋 Table of Contents

LIVE LINK - 

DATA SET LINK - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an automated pneumonia detection system using state-of-the-art deep learning techniques. The system can analyze chest X-ray images and classify them as either normal or showing signs of pneumonia, providing confidence scores and detailed probability breakdowns.

### Key Capabilities:
- **Real-time Analysis**: Instant classification of chest X-ray images
- **High Accuracy**: Trained on a large dataset of medical images
- **User-Friendly Interface**: Web-based application for easy access
- **Sample Images**: Built-in sample images for testing and demonstration
- **Professional Results**: Detailed probability breakdowns and confidence scores

## ✨ Features

### 🖥️ Streamlit Web Application
- **Interactive Upload**: Drag-and-drop or click-to-upload interface
- **Real-time Processing**: Instant analysis with loading indicators
- **Visual Results**: Side-by-side display of uploaded image and results
- **Sample Images**: Downloadable sample images for testing
- **Responsive Design**: Works on desktop and mobile devices
- **Professional UI**: Medical-themed interface with clear instructions

### 🤖 AI Model
- **YOLOv8 Classification**: State-of-the-art deep learning model
- **Binary Classification**: Normal vs. Pneumonia detection
- **Confidence Scoring**: Detailed probability breakdowns
- **Pre-trained Weights**: Ready-to-use trained model
- **Custom Training**: Full training pipeline included

### 📊 Results Display
- **Prediction Labels**: Clear normal/pneumonia classification
- **Confidence Metrics**: Percentage-based confidence scores
- **Probability Breakdown**: Detailed class-wise probabilities
- **Visual Indicators**: Color-coded results (green for normal, red for pneumonia)
- **Medical Disclaimers**: Professional healthcare warnings

  
## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Image-Processing
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import streamlit, ultralytics, PIL; print('Installation successful!')"
```

## 💻 Usage

### Running the Streamlit Application

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:8501`
   - The application will automatically load

3. **Using the Application**:
   - **Upload an Image**: Use the file uploader to select a chest X-ray image
   - **Download Samples**: If you don't have images, download the provided sample images
   - **View Results**: See the prediction results with confidence scores
   - **Analyze Probabilities**: Review the detailed probability breakdown

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 🎓 Model Training

### Training Data
The model is trained on a comprehensive dataset of chest X-ray images:
- **Normal Images**: 6,084 images of healthy chest X-rays
- **Pneumonia Images**: 7,744 images showing signs of pneumonia
- **Total Dataset**: 13,828 high-quality medical images

### Training Process
1. **Data Preparation**: Images are organized into normal and suffering categories
2. **Model Configuration**: YOLOv8 classification model with custom parameters
3. **Training Execution**: Automated training with validation
4. **Model Evaluation**: Performance metrics and confusion matrix generation
5. **Weight Saving**: Best and latest model weights are saved

### Training Script
```bash
cd yolov8_custom_training
python main.py
```

### Model Performance
- **Accuracy**: High classification accuracy on validation set
- **Confusion Matrix**: Visual representation of model performance
- **Training Metrics**: Detailed training and validation curves
- **Model Weights**: Optimized weights for production use

## 🌐 Streamlit Application Details

### Key Components

#### 1. **Model Loading**
- Cached model loading for performance
- Automatic error handling for missing model files
- Support for both best and latest weights

#### 2. **Image Processing**
- Automatic image format conversion
- Temporary file handling for YOLO compatibility
- Error handling for unsupported formats

#### 3. **Results Display**
- Two-column layout for image and results
- Color-coded prediction indicators
- Professional medical disclaimers
- Detailed probability breakdowns

#### 4. **Sample Images**
- Real training images for testing
- Downloadable sample files
- Authentic medical image quality

### Application Features
- **Responsive Design**: Works on all device sizes
- **Professional UI**: Medical-themed interface
- **Error Handling**: Graceful error messages
- **Loading States**: User-friendly loading indicators
- **File Validation**: Automatic format checking

## 📚 API Documentation

### Prediction Function
```python
def predict_image(image):
    """
    Make prediction on uploaded image
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: Prediction results with class names, probabilities, 
              predicted class, confidence, and class index
    """
```

### Model Loading
```python
@st.cache_resource
def load_model():
    """
    Load the trained YOLO model with caching
    
    Returns:
        YOLO: Loaded model object or None if error
    """
```

## ⚠️ Important Disclaimers

### Medical Disclaimer
This application is for **educational and demonstration purposes only**. It should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

### Model Limitations
- The model is trained on specific types of chest X-ray images
- Performance may vary with different image qualities and formats
- Results should be validated by medical professionals
- The system is not a replacement for professional medical diagnosis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the excellent object detection framework
- **Streamlit**: For the powerful web application framework
- **Medical Dataset**: Contributors to the chest X-ray dataset
- **Open Source Community**: For various supporting libraries and tools

**Built with ❤️ using Streamlit and YOLOv8**
