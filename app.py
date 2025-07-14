import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide"
)

# Title and description
st.title("🫁 Pneumonia Detector")
st.markdown("Upload a chest X-ray image to detect pneumonia or classify it as normal.")

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    try:
        model = YOLO('runs/classify/train/weights/last.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if the model file exists at 'runs/classify/train/weights/last.pt'")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Upload a chest X-ray image to detect pneumonia"
)

# Function to make prediction
def predict_image(image):
    """Make prediction on the uploaded image"""
    try:
        # Save the uploaded image temporarily
        temp_path = "temp_uploaded_image.jpg"
        image.save(temp_path, format='JPEG')
        
        # Make prediction using file path (same as predict.py)
        results = model(temp_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Get class names and probabilities
        names_dict = results[0].names
        probs_data = results[0].probs.data.tolist()
        
        # Get the predicted class
        predicted_class_idx = results[0].probs.top1
        predicted_class_name = names_dict[predicted_class_idx]
        predicted_confidence = results[0].probs.top1conf
        
        return {
            'class_names': names_dict,
            'probabilities': probs_data,
            'predicted_class': predicted_class_name,
            'confidence': predicted_confidence,
            'predicted_class_idx': predicted_class_idx
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Main application logic
if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🔍 Prediction Results")
        
        # Show loading spinner while processing
        with st.spinner("Analyzing image..."):
            # Make prediction
            result = predict_image(image)
        
        if result:
            # Display prediction results
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            class_names = result['class_names']
            probabilities = result['probabilities']
            
            # Create a more visually appealing result display
            if predicted_class == "normal":
                st.success(f"✅ **Prediction: {predicted_class.upper()}**")
                st.metric("Confidence", f"{confidence:.2%}")
            else:
                st.error(f"⚠️ **Prediction: {predicted_class.upper()}**")
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Display probability breakdown
            st.subheader("📊 Probability Breakdown")
            
            # Create a bar chart for probabilities
            prob_data = {
                'Class': list(class_names.values()),
                'Probability': [f"{prob:.2%}" for prob in probabilities]
            }
            
            # Display as a table
            import pandas as pd
            df = pd.DataFrame(prob_data)
            st.dataframe(df, use_container_width=True)
            
            # Add some styling based on prediction
            if predicted_class == "normal":
                st.info("🎉 The chest X-ray appears normal. No signs of pneumonia detected.")
            else:
                st.warning("🫁 Pneumonia detected. Please consult with a healthcare professional for proper diagnosis and treatment.")
    
    # Add some helpful information
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About this Pneumonia Detector
    
    This application uses a deep learning model trained on chest X-ray images to detect:
    - **Normal**: No signs of pneumonia
    - **Suffering**: Signs of pneumonia detected
    
    **Note**: This is for educational/demonstration purposes only. Always consult with healthcare professionals for medical diagnosis.
    """)

else:
    # Show placeholder when no image is uploaded
    st.info("👆 Please upload a chest X-ray image to get started!")
    
    # Sample images section
    st.markdown("---")
    st.subheader("📥 Sample Images for Testing")
    st.markdown("Don't have a chest X-ray image? Download these sample images to test the application:")
    
    # Create sample images using actual training data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample Normal X-ray**")
        # Load actual normal image from training data
        normal_image_path = "NORMAL2-IM-1427-0001.jpeg"
        
        try:
            normal_img = Image.open(normal_image_path)
            st.image(normal_img, caption="Sample Normal X-ray", use_container_width=True)
            
            # Create download button for normal sample
            with open(normal_image_path, "rb") as file:
                normal_data = file.read()
            
            st.download_button(
                label="📥 Download Normal Sample",
                data=normal_data,
                file_name="sample_normal_xray.jpeg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Error loading normal sample: {e}")
    
    with col2:
        st.markdown("**Sample Pneumonia X-ray**")
        # Load actual pneumonia image from training data
        pneumonia_image_path = "person1950_bacterial_4881.jpeg"
        
        try:
            pneumonia_img = Image.open(pneumonia_image_path)
            st.image(pneumonia_img, caption="Sample Pneumonia X-ray", use_container_width=True)
            
            # Create download button for pneumonia sample
            with open(pneumonia_image_path, "rb") as file:
                pneumonia_data = file.read()
            
            st.download_button(
                label="📥 Download Pneumonia Sample",
                data=pneumonia_data,
                file_name="sample_pneumonia_xray.jpeg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Error loading pneumonia sample: {e}")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Supported Image Formats
    - JPG/JPEG
    - PNG
    - BMP
    - TIFF
    
    ### 🎯 How it works
    1. Upload a chest X-ray image using the file uploader above (or download sample images)
    2. The AI model will analyze the image for signs of pneumonia
    3. Results will show the predicted class and confidence level
    4. A detailed probability breakdown will be displayed
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and YOLOv8*") 
