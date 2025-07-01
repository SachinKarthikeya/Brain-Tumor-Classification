# TumorScanAI — Brain Tumor Classification and Medical Chatbot

TumorScanAI is a deep learning-powered web application that performs **brain tumor classification from MRI images** using **AlexNet**, and provides **personalized medical assistance** via **Google Gemini-Pro chatbot**. The app is built with **PyTorch** for the model, **Streamlit** for the frontend, and integrates a **hospital locator** feature to connect users to nearby medical facilities.

## Features

- **Brain Tumor Classification** (Glioma, Meningioma, Pituitary, No Tumor) using AlexNet.
- **Emergency Assessment** with detailed explanations and recommended actions.
- **AI Chatbot Integration** using Google Gemini-Pro for medical Q&A and support.
- **Nearby Hospital Finder** based on user’s city or area.
- **Intuitive Interface** with real-time image upload and classification via Streamlit.

## Model Details

- **Architecture**: Modified **AlexNet** with the final layer adapted for 4 tumor classes.
- **Training Dataset**: Brain MRI dataset structured with `train/` and `val/` folders using `ImageFolder`.
- **Accuracy**: High classification performance on validation set with robust generalization.

## How It Works

1. User uploads an MRI image via the Streamlit interface.
2. The image is preprocessed and passed through the trained AlexNet model.
3. The model predicts the tumor type and assesses the emergency level.
4. An AI chatbot answers user queries and helps find nearby hospitals.

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- Streamlit
- Pillow
- python-dotenv
- Google Generative AI SDK (`google-generativeai`)
