import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
from docx import Document
import io

# 1. INITIALIZATION
st.set_page_config(page_title="EduDiagnostic Pro 2026", layout="wide", page_icon="📝")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key! Please add 'OPENAI_API_KEY' to Streamlit Secrets.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. THE BEAUTY LAYER: FORMATTER
def format_exam_paper(json_data):
    """Converts raw JSON into a professional, clean printed paper."""
    try:
        data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        
        # Header for the student paper
        paper = "STUDENT ASSESSMENT: " + st.session_state.get('last_lo', 'General') + "\n"
        paper += "NAME: __________________________    DATE: __________\n"
        paper += "="*60 + "\n\n"
        
        # Answer Key section (stored separately)
        key = "\n" + "="*20 + " TEACHER ANSWER KEY " + "="*20 + "\n"
        
        for i, q in enumerate(data.get('questions', []), 1):
            # Format Question
            paper += f"Question {i} ({q.get('level', 'Foundation')}):\n"
            paper += f"{q['question']}\n\n"
            
            # Format Options
            options = q.get('options', [])
            for idx, opt in enumerate(options):
                letter = chr(65 + idx) # A, B, C, D
                paper += f"   [{letter}] {opt}\n"
            
            paper += "\n" + "-"*40 + "\n\n"
            
            # Build Answer Key
            correct_idx = q.get('correct', 0)
            correct_letter = chr(65 + correct_idx)
            key += f"Q{i}: {correct_letter} | Misconception: {q.get('misconception_map', {}).get(str(correct_idx), 'N/A')}\n"
            
        return paper, key
    except Exception as e:
        return f"Formatting Error: {str(e)}", ""

# 3. DOWNLOAD UTILITIES (PDF & WORD)
def get_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    # Using multi_cell ensures text wraps and respects newlines
    pdf.multi_cell(0, 8, txt=text)
    return bytes(pdf.output())

def get_docx_bytes(text):
    doc = Document()
    doc.add_heading('Assessment Paper', 0)
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# 4. AGENT LOGIC
def agent_creator(lo, count):
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an Expert Exam Architect. Output clean JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# 5. USER INTERFACE
st.title("🎓 Professional Exam Generator & EI Engine")
st.info("Generates clean, printable assessments without brackets or code.")

with st.sidebar:
    st.header("Test Settings")
    lo = st.text_input("Learning Outcome", placeholder="e.g. Parts of a Plant")
    q_count = st.slider("Number of Questions", 5, 15, 5)
    st.session_state.last_lo = lo

if st.button("🚀 Generate Professional Assessment"):
    with st.spinner("Removing code brackets and formatting layout..."):
        raw_json = agent_creator(lo, q_count)
        student_paper, teacher_key = format_exam_paper(raw_json)
        
        # Store both in session state
        st.session_state.full_doc = student_paper + "\n\n" + teacher_key
        st.session_state.preview = student_paper
        st.success("Assessment Ready!")

if 'preview' in st.session_state:
    st.subheader("👀 Print Preview")
    st.text_area("Final Output", st.session_state.preview, height=400)
    
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Download PDF", 
                          data=get_pdf_bytes(st.session_state.full_doc), 
                          file_name="Assessment.pdf", mime="application/pdf")
    with c2:
        st.download_button("📥 Download Word (.docx)", 
                          data=get_docx_bytes(st.session_state.full_doc), 
                          file_name="Assessment.docx")
