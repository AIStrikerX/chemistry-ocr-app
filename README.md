 Smart Handwriting OCR with Llama 4

A Streamlit web application that converts handwritten chemistry notes into clean, structured Word documents using Groq’s Llama 4 Maverick multimodal vision model.

This project is designed to help students digitize handwritten chemistry notes while preserving formulas, equations, and logical structure.

 Live Application

 Streamlit App Link
https://chemistry-ocr-app-gtauljyy43rsnho5c7srpg.streamlit.app/

 Features

Llama 4 Maverick Integration
Fast and accurate multimodal understanding using Groq’s vision-capable LLM.

Chemistry-Focused Extraction
Preserves chemical formulas, reactions, symbols, and proportional relationships.

Structured Word Output
Automatically generates well-formatted .docx files with headings and bullet points.

Privacy-Focused Design
API keys are securely managed using Streamlit secrets and are never hardcoded.

 Tech Stack

Frontend: Streamlit

Model: Groq – Llama 4 Maverick (Vision)

Document Export: python-docx

Image Processing: OpenCV, Pillow

 Local Setup

Clone the repository

git clone <repository-url>
cd <repository-folder>


Install dependencies

pip install -r requirements.txt


Run the application

streamlit run app.py

 API Key Configuration

To run the app locally, create a Streamlit secrets file:

.streamlit/secrets.toml

GROQ_API_KEY = "your_key_here"


Your API key remains private and is not exposed in the source code.

☁️ Deployment (Streamlit Cloud)

This project is deployed using Streamlit Cloud, which is well-suited for Python-based ML applications.

Deployment steps:

Push the repository to GitHub

Connect the repository on Streamlit Cloud

Add GROQ_API_KEY under App Settings → Secrets

Deploy the app

 Notes

OCR accuracy depends on image clarity and handwriting quality

Diagrams are referenced textually rather than fully reconstructed

The app focuses on content accuracy and structure, not visual replication

 Use Case

Digitizing handwritten chemistry notes

Creating study-ready documents

Demonstrating multimodal AI for academic projects

Course or semester project submission
