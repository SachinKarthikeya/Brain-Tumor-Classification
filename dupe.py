import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Configure Streamlit page settings
st.set_page_config(
    page_title="TumorScanAI.com",
    page_icon=":brain:",
    layout="centered",
)
# Set up the brain tumor classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'alexnet_brain_tumor_classification.pth'

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

alexnet = models.alexnet(weights=None)  # Use 'weights=None' instead of 'pretrained=False'
alexnet.classifier[6] = nn.Linear(4096, 4)  # 4 classes: glioma, meningioma, notumor, pituitary
alexnet.load_state_dict(torch.load(model_path, map_location=device))
alexnet = alexnet.to(device)
alexnet.eval()

class_names = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

# Function to predict the class of an uploaded image
def predict_image(image, model):
    image = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Function to assess emergency level and suggest treatment or vitamins
def assess_tumor_emergency(predicted_class):
    emergency_info = {
        "Glioma": {
            "emergency_level": "High",
            "action": "You should immediately see a doctor. Glioma can be aggressive and requires professional treatment.",
            "origin": "Arises from glial cells in the brain and spinal cord."
        },
        "Meningioma": {
            "emergency_level": "Moderate",
            "action": "Consult with a doctor soon. Meningioma is usually slow-growing but requires medical supervision.",
            "origin": "Arises from the meninges, which cover the brain and spinal cord."
        },
        "Notumor": {
            "emergency_level": "None",
            "action": "No tumor detected. Maintain a healthy lifestyle and monitor for symptoms."
        },
        "Pituitary": {
            "emergency_level": "Medium",
            "action": "Visit a doctor to discuss treatment options. Pituitary tumors can affect hormone levels and require medical attention.",
            "origin": "Arises from the pituitary gland, responsible for hormone production."
        }
    }
    return emergency_info.get(predicted_class, {"emergency_level": "Unknown", "action": "Consult a medical professional."})

# Main Streamlit App
def main():
    # Add a heading for the website
    st.markdown("<h1 style='text-align: center;'>Brain Tumor Classification</h1>", unsafe_allow_html=True)
    
    # Brain Tumor Classification Section
    st.header("Upload an MRI Image for Classification")
    uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((300, 300))
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(resized_image, caption="Uploaded MRI Image", use_column_width=True)
        
        with col2:
            if st.button("Classify Tumor"):
                predicted_class = predict_image(image, alexnet)
                
                # Display classification result
                st.markdown(f"<h2>The predicted class is: {predicted_class}</h2>", unsafe_allow_html=True)

                # Assess emergency level and suggested action
                assessment = assess_tumor_emergency(predicted_class)
                st.markdown(f"**Emergency Level**: {assessment['emergency_level']}")
                st.markdown(f"**Recommended Action**: {assessment['action']}")
                
                if 'origin' in assessment:
                    st.markdown(f"**Origin**: {assessment['origin']}")

    else:
        st.warning("Please upload an MRI image for classification.")

if __name__ == "__main__":
    main()
