import os
import streamlit as st
from dotenv import load_dotenv
import requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import ollama

load_dotenv()

st.set_page_config(
    page_title="TumorScanAI.com",
    page_icon=":brain:",
    layout="centered",
)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# SerpAPI function to search hospitals
def search_hospitals_with_serpapi(location):
    params = {
        "engine": "google",
        "q": f"hospitals in {location}",
        "api_key": SERPAPI_API_KEY,
        "num": 5
    }
    response = requests.get(SERPAPI_URL, params=params)
    results = response.json()
    if "organic_results" in results:
        hospitals = []
        for result in results["organic_results"][:5]:
            title = result.get("title")
            link = result.get("link")
            snippet = result.get("snippet", "")
            hospitals.append(f"**{title}**\n\n{snippet}\n[Visit Website]({link})")
        return hospitals
    return ["Sorry, no hospital information could be found."]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'alexnet_brain_tumor_classification.pth'

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

alexnet = models.alexnet(pretrained=False)
alexnet.classifier[6] = nn.Linear(4096, 4)
alexnet.load_state_dict(torch.load(model_path, map_location=device))
alexnet = alexnet.to(device)
alexnet.eval()

class_names = ['Glioma', 'Meningioma', 'Notumor', 'Pituitary']

def predict_image(image, model):
    image = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

def autoresponder(prompt):
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

def main():
    predicted_class = None
    st.markdown("<h1 style='text-align: center;'>Brain Tumor Classification and BrainyBot</h1>", unsafe_allow_html=True)

    st.header("Brain Tumor Classification")
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
                st.session_state.predicted_class = predicted_class
                classification_result = f"The predicted class is: {predicted_class}"

                st.markdown(f"<h2>{classification_result}</h2>", unsafe_allow_html=True)

                if predicted_class == "Glioma":
                    prompt = """
                    User has uploaded an MRI image and the predicted class is Glioma.
                    Please provide a detailed response regarding the emergency level, recommended action, and origin of the tumor.
                    Please provide an educational summary about this tumor: its emergency level, treatment recommendation, and origin. Do not include medical disclaimers. Keep the explanation informative and concise. 
                    """
                    with st.spinner("Gathering information..."):
                        response = autoresponder(prompt)
                        st.markdown(f"<h3>Tumor Info:</h3><p>{response}</p>", unsafe_allow_html=True)

                elif predicted_class == "Meningioma":
                    prompt = """
                    User has uploaded an MRI image and the predicted class is Meningioma.
                    Please provide a detailed response regarding the emergency level, recommended action, and origin of the tumor.
                    Please provide an educational summary about this tumor: its emergency level, treatment recommendation, and origin. Do not include medical disclaimers. Keep the explanation informative and concise.
                    """
                    with st.spinner("Gathering information..."):
                        response = autoresponder(prompt)
                        st.markdown(f"<h3>Tumor Info:</h3><p>{response}</p>", unsafe_allow_html=True)

                elif predicted_class == "Pituitary":
                    prompt = """
                    User has uploaded an MRI image and the predicted class is Pituitary.
                    Please provide a detailed response regarding the emergency level, recommended action, and origin of the tumor.
                    Please provide an educational summary about this tumor: its emergency level, treatment recommendation, and origin. Do not include medical disclaimers. Keep the explanation informative and concise.
                    """
                    with st.spinner("Gathering information..."):
                        response = autoresponder(prompt)
                        st.markdown(f"<h3>Tumor Info:</h3><p>{response}</p>", unsafe_allow_html=True)

                else:
                    st.success("No tumor detected. You may continue a healthy lifestyle, but always monitor your health regularly.")
    else:
        st.warning("Please upload an MRI image for classification.")

    st.header("Find Nearby Hospitals")
    user_location = st.text_input("Enter your area or city to find nearby hospitals:")

    if user_location:
        st.markdown("🔍 Searching for nearby hospitals...")
        hospitals_list = search_hospitals_with_serpapi(user_location)
        for hospital_info in hospitals_list:
            st.markdown("---")
            st.markdown(hospital_info, unsafe_allow_html=True)

if __name__ == "__main__":
    main()