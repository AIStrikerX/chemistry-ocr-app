import streamlit as st
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import tempfile
import cv2
import numpy as np
import os
import ssl
import re
from datetime import datetime
try:
    import streamlit.secret as secrets
except ImportError:
    pass

# WORKAROUND: Bypass SSL verification for model downloads if certificates are missing
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set page configuration
st.set_page_config(page_title="Smart OCR - Chemistry Notes", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button { color: white; background-color: #4CAF50; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Smart OCR - Chemistry Notes")
st.markdown("**Layout-aware OCR** for handwritten chemistry notes with structured Word export")

def init_agent_state():
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = {
            "shorthand": {},
            "corrections": [],
            "processed_documents": 0
        }
    if "agent_safety_logs" not in st.session_state:
        st.session_state.agent_safety_logs = []

init_agent_state()

# Sidebar for engine selection
st.sidebar.title("⚙️ OCR Settings")
ocr_engine = st.sidebar.radio(
    "Select OCR Engine:",
    ["Groq (Llama 4 Maverick)", "Hugging Face (SambaNova)", "Gemini AI (Best Quality)"],
    help="Groq Llama 4 is fast and free (preview). Hugging Face uses SambaNova provider for Llama 4 Maverick. Gemini requires API key."
)

# Default transcription mode (overridden when Groq or HF is selected)
transcription_mode = "Relaxed (Clean Notes)"

# API Keys
gemini_api_key = None
groq_api_key = None
hf_api_key = None

if ocr_engine == "Gemini AI (Best Quality)":
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key:",
        type="password",
        help="Get your free key at: https://aistudio.google.com/app/apikey"
    )
    if not gemini_api_key:
        st.sidebar.warning("⚠️ API key required for Gemini")

elif ocr_engine == "Hugging Face (SambaNova)":
    # Try to load from secrets first
    if "HF_TOKEN" in st.secrets:
        hf_api_key = st.secrets["HF_TOKEN"]
    else:
        hf_api_key = st.sidebar.text_input(
            "Hugging Face Token:",
            type="password",
            help="Get your key at: https://huggingface.co/settings/tokens"
        )
    
    transcription_mode = st.sidebar.radio(
        "Transcription Mode:",
        ["Relaxed (Clean Notes)", "Strict (Exact Transcription)"],
        help="Relaxed fixes minor errors. Strict keeps text exactly as written."
    )
    
    if not hf_api_key:
        st.sidebar.warning("⚠️ API key required for Hugging Face")

elif ocr_engine == "Groq (Llama 4 Maverick)":
    # Try to load from secrets first
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_api_key = st.sidebar.text_input(
            "Groq API Key:",
            type="password",
            help="Get your free key at: https://console.groq.com/keys"
        )
    
    transcription_mode = st.sidebar.radio(
        "Transcription Mode:",
        ["Relaxed (Clean Notes)", "Strict (Exact Transcription)"],
        help="Relaxed fixes minor errors. Strict keeps text exactly as written."
    )
    
    if not groq_api_key:
        st.sidebar.warning("⚠️ API key required for Groq")

st.sidebar.markdown("---")
st.sidebar.title("🤖 Phase 2 Agentic Mode")
agentic_mode = st.sidebar.checkbox(
    "Enable Smart Chemistry Agent",
    value=True,
    help="Adds perception, decision support, memory, safety logging, and human-in-the-loop review."
)
agent_autonomy = st.sidebar.select_slider(
    "Agent Autonomy Level:",
    options=["Manual Review", "Semi-Autonomous"],
    value="Semi-Autonomous",
    help="Semi-autonomous mode suggests actions, but final document approval remains with the user."
)
enable_memory = st.sidebar.checkbox(
    "Enable Session Memory",
    value=True,
    help="Stores corrections and shorthand only in this browser session."
)
enable_safety_log = st.sidebar.checkbox(
    "Enable Safety Log",
    value=True,
    help="Adds agent decisions, risks, and human approval status to the Word document."
)


# Initialize engines
@st.cache_resource
def load_groq_client(api_key):
    from groq import Groq
    return Groq(api_key=api_key)

@st.cache_resource
def load_huggingface_client(api_key):
    from huggingface_hub import InferenceClient
    # provider="sambanova" routes requests to SambaNova's hosted Llama 4 Maverick
    return InferenceClient(provider="sambanova", api_key=api_key)

@st.cache_resource
def load_gemini(_api_key):
    import google.generativeai as genai
    genai.configure(api_key=_api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def groq_ocr_process(image_path, api_key, mode="Relaxed (Clean Notes)"):
    """Process image with Groq Llama 4 Maverick using improved system prompt"""
    try:
        client = load_groq_client(api_key)
        base64_image = encode_image(image_path)
        
        # Base System Prompt
        system_prompt = """You are an expert scientific document transcriber and editor.

Your task is to convert handwritten chemistry notes from an image into clean, structured study notes.

IMPORTANT CONTEXT:
- The source is a photographed handwritten notebook page.
- The page may contain multiple columns, boxed sections, underlines, arrows, diagrams, equations, and headings written in different colors.
- Some content is textual, some is visual (diagrams, flowcharts, graphs).

YOUR RESPONSIBILITIES:
1. Accurately transcribe ALL readable text from the image.
2. Preserve the original meaning, terminology, and scientific correctness.
3. Reconstruct a logical structure similar to well-written chemistry notes.

STRUCTURE RULES:
- Use clear section headings for major topics.
- Use subheadings where appropriate.
- Use bullet points or numbered lists when the content implies lists.
- Maintain logical reading order (top-to-bottom, left-to-right).
- Do NOT invent new content or explanations.

EQUATIONS & SYMBOLS:
- Preserve chemical symbols, formulas, charges, arrows, and proportionality signs.
- Use LaTeX-style inline math only when necessary (e.g., H₂SO₄, V₂O₅, 1/viscosity).
- Do NOT "correct" chemistry unless the handwriting is clearly ambiguous.

DIAGRAMS & VISUAL ELEMENTS:
- If a diagram, graph, flowchart, or visual illustration is present:
  - Do NOT attempt to recreate it as text.
  - Insert a placeholder in the format:
    [DIAGRAM: short factual description of what is shown]
- If arrows indicate process flow, reflect that flow in text where obvious.

QUALITY CONTROL:
- If a word is unclear, transcribe the closest plausible chemistry term without guessing new concepts.
- Do not paraphrase or summarize aggressively.
- Do not beautify language; keep it note-like and concise.

OUTPUT FORMAT:
- Output clean, structured Markdown.
- Do NOT include explanations about what you are doing.
- Do NOT mention OCR, AI, or the model.
- Output ONLY the final structured notes."""

        # Add Mode Logic
        if "Strict" in mode:
            system_prompt += "\n\nSTRICT MODE: If any text is unclear, preserve it as written rather than guessing."
        else:
            system_prompt += "\n\nRELAXED MODE: Minor spelling corrections are allowed only for standard chemistry terms."

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe and structure the handwritten chemistry notes in this image according to the system instructions."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.2, # Added reliability constraint
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq Error: {e}")
        return None

def huggingface_ocr_process(image_path, api_key, mode="Relaxed (Clean Notes)"):
    """Process image with Hugging Face InferenceClient using SambaNova provider"""
    try:
        client = load_huggingface_client(api_key)
        base64_image = encode_image(image_path)
        
        # Base System Prompt
        system_prompt = """You are an expert scientific document transcriber and editor.

Your task is to convert handwritten chemistry notes from an image into clean, structured study notes.

IMPORTANT CONTEXT:
- The source is a photographed handwritten notebook page.
- The page may contain multiple columns, boxed sections, underlines, arrows, diagrams, equations, and headings written in different colors.
- Some content is textual, some is visual (diagrams, flowcharts, graphs).

YOUR RESPONSIBILITIES:
1. Accurately transcribe ALL readable text from the image.
2. Preserve the original meaning, terminology, and scientific correctness.
3. Reconstruct a logical structure similar to well-written chemistry notes.

STRUCTURE RULES:
- Use clear section headings for major topics.
- Use subheadings where appropriate.
- Use bullet points or numbered lists when the content implies lists.
- Maintain logical reading order (top-to-bottom, left-to-right).
- Do NOT invent new content or explanations.

EQUATIONS & SYMBOLS:
- Preserve chemical symbols, formulas, charges, arrows, and proportionality signs.
- Use LaTeX-style inline math only when necessary (e.g., H₂SO₄, V₂O₅, 1/viscosity).
- Do NOT "correct" chemistry unless the handwriting is clearly ambiguous.

DIAGRAMS & VISUAL ELEMENTS:
- If a diagram, graph, flowchart, or visual illustration is present:
  - Do NOT attempt to recreate it as text.
  - Insert a placeholder in the format:
    [DIAGRAM: short factual description of what is shown]
- If arrows indicate process flow, reflect that flow in text where obvious.

QUALITY CONTROL:
- If a word is unclear, transcribe the closest plausible chemistry term without guessing new concepts.
- Do not paraphrase or summarize aggressively.
- Do not beautify language; keep it note-like and concise.

OUTPUT FORMAT:
- Output clean, structured Markdown.
- Do NOT include explanations about what you are doing.
- Do NOT mention OCR, AI, or the model.
- Output ONLY the final structured notes."""

        # Add Mode Logic
        if "Strict" in mode:
            system_prompt += "\n\nSTRICT MODE: If any text is unclear, preserve it as written rather than guessing."
        else:
            system_prompt += "\n\nRELAXED MODE: Minor spelling corrections are allowed only for standard chemistry terms."

        # SambaNova provider supports Llama-4-Maverick vision via InferenceClient
        completion = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe and structure the handwritten chemistry notes in this image according to the system instructions."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Hugging Face Error: {e}")
        return None

def preprocess_image(image_path):
    """Enhanced preprocessing for better OCR"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

def gemini_ocr_process(image_path, api_key):
    """Process image with Gemini Vision API"""
    try:
        model = load_gemini(api_key)
        img = Image.open(image_path)
        
        prompt = """You are an expert transcriber for handwritten chemistry notes.

Please transcribe this handwritten chemistry note into clean, structured Markdown format.

Follow these rules:
1. Preserve the document structure (headings, paragraphs, lists)
2. Use proper Markdown headings (# for main topics, ## for subtopics)
3. Format chemical equations properly
4. If you see diagrams or complex equations, describe them as: `[DIAGRAM: brief description]`
5. Maintain the reading order (left-to-right, top-to-bottom)
6. Fix any obvious spelling errors in chemistry terms

Output ONLY the Markdown text, no additional commentary."""

        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None

def extract_chemical_candidates(text):
    if not text:
        return []
    pattern = r'\b(?:[A-Z][a-z]?\d*){2,}(?:[+-])?\b'
    candidates = sorted(set(re.findall(pattern, text)))
    return candidates[:20]

def detect_agentic_risks(text):
    risks = []
    if not text or len(text.strip()) < 30:
        risks.append({
            "level": "High",
            "issue": "Very low extracted content",
            "action": "Ask the user to upload a clearer image or retry with another OCR engine."
        })
    if any(token in text.lower() for token in ["unclear", "illegible", "not readable", "?"]):
        risks.append({
            "level": "Medium",
            "issue": "Unclear handwriting detected",
            "action": "Require human review before finalizing the document."
        })
    if any(symbol in text for symbol in ["→", "->", "⇌", "="]):
        risks.append({
            "level": "Medium",
            "issue": "Chemical equation or reaction detected",
            "action": "Flag formulas for manual verification before use in study material."
        })
    if "[DIAGRAM" in text:
        risks.append({
            "level": "Low",
            "issue": "Diagram placeholder detected",
            "action": "Embed original image as reference because diagrams are not fully reconstructed."
        })
    if not risks:
        risks.append({
            "level": "Low",
            "issue": "No major risk detected",
            "action": "Proceed with normal human review."
        })
    return risks

def build_agentic_review(ocr_result, engine_type, autonomy_level, memory_enabled):
    chemical_candidates = extract_chemical_candidates(ocr_result)
    risks = detect_agentic_risks(ocr_result)
    confidence = "High"
    if any(risk["level"] == "High" for risk in risks):
        confidence = "Low"
    elif any(risk["level"] == "Medium" for risk in risks):
        confidence = "Medium"
    review = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "engine": engine_type.upper(),
        "autonomy_level": autonomy_level,
        "confidence": confidence,
        "chemical_candidates": chemical_candidates,
        "risks": risks,
        "agent_decision": "Human approval required before final DOCX export.",
        "memory_status": "Session memory enabled" if memory_enabled else "Session memory disabled"
    }
    if autonomy_level == "Semi-Autonomous" and confidence == "High":
        review["agent_decision"] = "Safe to generate DOCX after user confirmation."
    return review

def render_agentic_review(review):
    st.metric("Agent Confidence", review["confidence"])
    st.write(f"**OCR Engine:** {review['engine']}")
    st.write(f"**Autonomy Level:** {review['autonomy_level']}")
    st.write(f"**Agent Decision:** {review['agent_decision']}")
    if review["chemical_candidates"]:
        st.write("**Detected Chemical Candidates:**")
        st.write(", ".join(review["chemical_candidates"]))
    else:
        st.info("No clear chemical formula candidates detected by the rule-based scanner.")
    st.write("**Risk Assessment:**")
    for risk in review["risks"]:
        if risk["level"] == "High":
            st.error(f"{risk['level']}: {risk['issue']} — {risk['action']}")
        elif risk["level"] == "Medium":
            st.warning(f"{risk['level']}: {risk['issue']} — {risk['action']}")
        else:
            st.success(f"{risk['level']}: {risk['issue']} — {risk['action']}")

def apply_memory_to_text(text):
    if not enable_memory:
        return text
    updated_text = text
    for shorthand, expansion in st.session_state.agent_memory["shorthand"].items():
        updated_text = re.sub(rf'\b{re.escape(shorthand)}\b', expansion, updated_text)
    return updated_text

def add_agentic_report_to_doc(doc, agent_report, human_approved, memory_snapshot):
    doc.add_page_break()
    doc.add_heading("Phase 2 Agentic Review & Safety Log", level=1)
    doc.add_paragraph(f"Review Timestamp: {agent_report['timestamp']}")
    doc.add_paragraph(f"OCR Engine: {agent_report['engine']}")
    doc.add_paragraph(f"Autonomy Level: {agent_report['autonomy_level']}")
    doc.add_paragraph(f"Agent Confidence: {agent_report['confidence']}")
    doc.add_paragraph(f"Human Approval: {'Approved' if human_approved else 'Not Approved'}")
    doc.add_paragraph(f"Memory Status: {agent_report['memory_status']}")
    doc.add_heading("Detected Chemical Candidates", level=2)
    if agent_report["chemical_candidates"]:
        for candidate in agent_report["chemical_candidates"]:
            doc.add_paragraph(candidate, style='List Bullet')
    else:
        doc.add_paragraph("No rule-based formula candidates detected.")
    doc.add_heading("Risk Assessment", level=2)
    for risk in agent_report["risks"]:
        doc.add_paragraph(f"{risk['level']}: {risk['issue']} — {risk['action']}", style='List Bullet')
    doc.add_heading("Session Memory Snapshot", level=2)
    if memory_snapshot and memory_snapshot.get("shorthand"):
        for shorthand, expansion in memory_snapshot["shorthand"].items():
            doc.add_paragraph(f"{shorthand} → {expansion}", style='List Bullet')
    else:
        doc.add_paragraph("No shorthand memory saved in this session.")

def create_structured_docx(ocr_result, image_path, engine_type="groq", agent_report=None, human_approved=False, memory_snapshot=None):
    """Create a structured Word document from OCR results"""
    doc = Document()
    
    # Title
    title = doc.add_heading("Chemistry Notes - OCR Extraction", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph(f"OCR Engine: {engine_type.upper()}")
    doc.add_paragraph("_" * 50)
    
    if engine_type in ["groq", "gemini", "huggingface"]:
        # Markdown output - convert to Word
        if ocr_result:
            lines = ocr_result.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect markdown headings
                if line.startswith('### '):
                    doc.add_heading(line[4:], level=3)
                elif line.startswith('## '):
                    doc.add_heading(line[3:], level=2)
                elif line.startswith('# '):
                    doc.add_heading(line[2:], level=1)
                elif line.startswith('- ') or line.startswith('* '):
                    doc.add_paragraph(line[2:], style='List Bullet')
                elif line.startswith('[DIAGRAM'):
                    doc.add_paragraph(line, style='Intense Quote')
                else:
                    doc.add_paragraph(line)
    
    # Add separator
    doc.add_paragraph()
    doc.add_paragraph("_" * 50)
    
    # Add original image at the end for reference
    doc.add_heading("Original Image (Reference)", level=2)
    try:
        doc.add_picture(image_path, width=Inches(6))
    except:
        doc.add_paragraph("[Could not embed original image]")

    if agent_report:
        add_agentic_report_to_doc(doc, agent_report, human_approved, memory_snapshot)
    
    return doc


# Main app
uploaded_file = st.file_uploader("📤 Upload handwritten note", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    doc = None
    ocr_result = None
    engine_type = None
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(input_path, use_container_width=True)
    
    with col2:
        st.subheader("📝 Extracted Content")
        
        if ocr_engine == "Groq (Llama 4 Maverick)":
            if groq_api_key:
                with st.spinner('⚡ Groq (Llama 4 Maverick) is analyzing your note...'):
                    ocr_result = groq_ocr_process(input_path, groq_api_key, transcription_mode)
                    
                    if ocr_result:
                        engine_type = "groq"
                    else:
                        st.error("Failed to process with Groq. Check API Key or try again.")
            else:
                st.warning("⚠️ Please enter your Groq API key in the sidebar")
        
        elif ocr_engine == "Hugging Face (SambaNova)":
            if hf_api_key:
                with st.spinner('🤖 Hugging Face (SambaNova) is analyzing your note...'):
                    ocr_result = huggingface_ocr_process(input_path, hf_api_key, transcription_mode)
                    
                    if ocr_result:
                        engine_type = "huggingface"
                    else:
                        st.error("Failed to process with Hugging Face. Check API Key or try again.")
            else:
                st.warning("⚠️ Please enter your Hugging Face token in the sidebar")
            
        else:  # Gemini
            if gemini_api_key:
                with st.spinner('🤖 Gemini AI is analyzing your note...'):
                    ocr_result = gemini_ocr_process(input_path, gemini_api_key)
                    
                    if ocr_result:
                        engine_type = "gemini"
                    else:
                        st.error("Failed to process with Gemini")
            else:
                st.warning("⚠️ Please enter your Gemini API key in the sidebar")

        if ocr_result:
            if agentic_mode and enable_memory:
                ocr_result = apply_memory_to_text(ocr_result)
            st.markdown(ocr_result)

    if ocr_result and engine_type:
        agent_report = None
        human_approved = True
        reviewed_text = ocr_result

        if agentic_mode:
            st.markdown("---")
            st.subheader("🤖 Phase 2 Agentic Review")
            st.caption("Observe → Interpret → Decide → Act → Learn with human-in-the-loop approval.")
            agent_report = build_agentic_review(ocr_result, engine_type, agent_autonomy, enable_memory)
            review_col, memory_col = st.columns([2, 1])

            with review_col:
                render_agentic_review(agent_report)
                human_approved = st.checkbox(
                    "✅ I reviewed the extracted content and approve this document for export",
                    value=agent_report["confidence"] == "High"
                )
                reviewed_text = st.text_area(
                    "Human-in-the-loop editable final text",
                    value=ocr_result,
                    height=260,
                    help="Correct OCR mistakes here before the final Word document is generated."
                )

            with memory_col:
                st.markdown("### 🧠 Session Memory")
                st.write(f"Processed documents: {st.session_state.agent_memory['processed_documents']}")
                shorthand = st.text_input("Shorthand", placeholder="rxn")
                expansion = st.text_input("Expansion", placeholder="reaction")
                if st.button("Save shorthand to memory", use_container_width=True):
                    if shorthand.strip() and expansion.strip():
                        st.session_state.agent_memory["shorthand"][shorthand.strip()] = expansion.strip()
                        st.success("Saved in session memory. Re-process or edit final text to apply.")
                    else:
                        st.warning("Enter both shorthand and expansion.")
                if st.button("Clear session memory", use_container_width=True):
                    st.session_state.agent_memory["shorthand"].clear()
                    st.session_state.agent_memory["corrections"].clear()
                    st.success("Session memory cleared.")
                if st.session_state.agent_memory["shorthand"]:
                    st.write("**Saved shorthand:**")
                    for shorthand_key, expansion_value in st.session_state.agent_memory["shorthand"].items():
                        st.write(f"- `{shorthand_key}` → `{expansion_value}`")

            if enable_safety_log:
                st.session_state.agent_safety_logs.append(agent_report)
            if enable_memory and reviewed_text != ocr_result:
                st.session_state.agent_memory["corrections"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "engine": engine_type,
                    "before_length": len(ocr_result),
                    "after_length": len(reviewed_text)
                })

        memory_snapshot = {
            "shorthand": dict(st.session_state.agent_memory["shorthand"]),
            "corrections": list(st.session_state.agent_memory["corrections"])
        }
        doc = create_structured_docx(
            reviewed_text,
            input_path,
            engine_type=engine_type,
            agent_report=agent_report if agentic_mode and enable_safety_log else None,
            human_approved=human_approved,
            memory_snapshot=memory_snapshot
        )
        st.session_state.agent_memory["processed_documents"] += 1
    
    # Download button
    if doc is not None:
        doc_path = "chemistry_notes_ocr.docx"
        doc.save(doc_path)
        
        with open(doc_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Structured Word Document",
                data=f,
                file_name="chemistry_notes_structured.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        
        # Cleanup
        try:
            os.remove(input_path)
        except:
            pass

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Tips for Best Results
- **Lighting**: Ensure good, even lighting
- **Focus**: Clear, sharp images work best
- **Resolution**: Higher resolution = better accuracy
- **Groq Llama 4**: Extremely fast, multimodal
- **Gemini**: Excellent for complex layouts

### 🧪 Chemistry Notes
- Diagrams will be marked as `[DIAGRAM]`
- Equations are formatted in Markdown
- Original image embedded for reference
""")
