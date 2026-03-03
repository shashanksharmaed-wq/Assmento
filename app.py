import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
from docx import Document
import io
import re

# 1. INITIALIZATION
st.set_page_config(page_title="EduDiagnostic Pro 2026", layout="wide", page_icon="📝")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key! Please add 'OPENAI_API_KEY' to Streamlit Secrets.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. THE "NO-BRACKETS" CLEANING ENGINE
def scrub_to_human_text(json_input):
    """Specifically removes all JSON/HTML artifacts and builds a clean paper."""
    try:
        # If input is string, convert to dict
        data = json.loads(json_input) if isinstance(json_input, str) else json_input
        
        # Build the Human-Readable String
        clean_paper = f"ASSESSMENT: {st.session_state.get('last_lo', 'General Topic')}\n"
        clean_paper += "STUDENT NAME: __________________________    DATE: __________\n"
        clean_paper += "="*60 + "\n\n"
        
        questions = data.get('questions', [])
        for i, q in enumerate(questions, 1):
            clean_paper += f"QUESTION {i} [{q.get('level', 'Foundation')}]:\n"
            clean_paper += f"{q.get('question', 'No question text provided.')}\n\n"
            
            opts = q.get('options', [])
            for idx, opt in enumerate(opts):
                clean_paper += f"   {chr(65+idx)}) {opt}\n"
            
            clean_paper += "\n" + "."*40 + "\n\n"
            
        # Add Answer Key at the very end
        clean_paper += "\n\n" + "="*20 + " TEACHER'S ANSWER KEY " + "="*20 + "\n"
        for i, q in enumerate(questions, 1):
            correct_idx = q.get('correct', 0)
            clean_paper += f"Q{i}: {chr(65+correct_idx)} | Concept: {q.get('level')}\n"
            
        return clean_paper
    except Exception as e:
        # Fallback: if JSON fails, manually strip brackets using Regex
        text = str(json_input)
        text = re.sub(r'[{} ["[\]]', '', text) # Remove brackets/braces
        return f"Formatting Error, but here is the cleaned text:\n\n{text}"

# 3. FILE GENERATORS
def get_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 7, txt=text)
    return bytes(pdf.output())

def get_docx_bytes(text):
    doc = Document()
    doc.add_heading('Classroom Assessment', 0)
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# 4. AGENT LOGIC
def agent_creator(lo, count):
    # Strict prompt to ensure JSON is returned for our cleaner to process
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. Return ONLY a JSON object with a 'questions' list."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a professional exam formatter. Output valid JSON."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# 5. UI
st.title("🎓 Professional Diagnostic Engine")
st.markdown("### Step 1: Generate a Clean Paper")

with st.sidebar:
    lo = st.text_input("Learning Outcome", "Human Digestive System")
    q_count = st.slider("Number of Questions", 5, 15, 5)
    st.session_state.last_lo = lo

if st.button("🚀 Generate Print-Ready Assessment"):
    with st.spinner("Removing brackets and building layout..."):
        raw_data = agent_creator(lo, q_count)
        # FORCE THE CLEANING
        st.session_state.clean_output = scrub_to_human_text(raw_data)
        st.success("Clean Assessment Created!")

if 'clean_output' in st.session_state:
    st.subheader("📝 Live Preview (Brackets Removed)")
    st.text_area("Final Version", st.session_state.clean_output, height=400)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download PDF", data=get_pdf_bytes(st.session_state.clean_output), file_name="Test.pdf")
    with col2:
        st.download_button("📥 Download Word", data=get_docx_bytes(st.session_state.clean_output), file_name="Test.docx")
