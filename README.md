# Smart Handwriting OCR with Llama 4

A Streamlit application that converts handwritten chemistry notes into structured Word documents using Groq's Llama 4 Maverick vision model.

## Features
- **Llama 4 Maverick Integration**: Fast, accurate multimodal understanding.
- **Chemistry Specialized**: Preserves consistency in formulas, equations, and diagrams.
- **Structured Output**: Exports directly to `.docx` with proper headings and formatting.
- **Privacy Focused**: API keys are handled securely via Streamlit secrets.

## Setup

1.  **Clone the repository**
2.  **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## Secrets

To run locally, create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_key_here"
```

## Deployment

Deploy directly to **Streamlit Cloud**:
1.  Push to GitHub.
2.  Connect repository in Streamlit Cloud.
3.  Add `GROQ_API_KEY` in Streamlit Cloud's "Advanced Settings" -> "Secrets".
