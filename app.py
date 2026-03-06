import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io
import time

# --- 1. BRANDING & AUTHENTICATION ---
st.set_page_config(page_title="Assemento Elite 2026", layout="wide", page_icon="🎯")
USER_DB = {f"T{i}": f"T{1233+i}" for i in range(1, 51)}

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🎯 Assemento Elite: Private Beta")
    with st.form("login_form"):
        u_id = st.text_input("Teacher ID (T1-T50)")
        p_wd = st.text_input("Password", type="password")
        if st.form_submit_button("Enter Engine"):
            if USER_DB.get(u_id) == p_wd:
                st.session_state.authenticated = True
                st.session_state.current_user = u_id
                st.rerun()
            else:
                st.error("Invalid Credentials.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 2. THE MCQ ASSESSMENT PDF ENGINE (Structured Grid) ---
class AssementoExamPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 15, f'ASSEMENTO ASSESSMENT - {st.session_state.get("current_user")}', 0, 1, 'C')
        self.ln(10)

def get_assessment_pdf(questions, title, aid):
    pdf = AssementoExamPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Paper Header
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"TOPIC: {title.upper()}", ln=True, align='C')
    pdf.set_font("helvetica", "I", 9)
    pdf.cell(0, 5, f"Assessment Fingerprint: {aid}", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 10, "Student Name: __________________________   Date: __________", ln=True)
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    for i, q in enumerate(questions, 1):
        # Question Stem (Multi-line support)
        pdf.set_font("helvetica", "B", 11)
        pdf.multi_cell(0, 7, txt=f"Q{i}. {q['question']}")
        pdf.ln(2)
        
        # Options Block (Indented A, B, C, D)
        pdf.set_font("helvetica", "", 10)
        for j, opt in enumerate(q['options']):
            prefix = chr(65 + j) # A, B, C, D
            pdf.set_x(20) # Indent options
            pdf.cell(0, 7, txt=f"{prefix}) {opt}", ln=True)
        
        pdf.ln(6) # Spacing between questions
        if pdf.get_y() > 250: # Avoid orphans at page bottom
            pdf.add_page()
            
    return bytes(pdf.output())

# --- 3. ASSEMENTO ASSESSMENT CREATOR (Agent 1) ---
def agent_assessment_creator(lo, count, tiers):
    aid = f"AID-{st.session_state.current_user}-{int(time.time())}"
    # Strict JSON formatting instruction
    prompt = f"""
    Create a {count}-question Multiple Choice Question (MCQ) test for LO: {lo}. 
    Difficulty Tiers: {tiers}. 
    Each question must have exactly 4 options.
    
    OUTPUT ONLY VALID JSON:
    {{
      "questions": [
        {{
          "question": "The question text here?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "correct_index": 0,
          "level": "Foundation"
        }}
      ]
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator. Output JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), aid

# --- 4. UI INTERFACE ---
st.title(f"🚀 Assemento Engine: {st.session_state.current_user}")
tab1, tab2 = st.tabs(["🏗️ Creator", "📊 Diagnostic Engine"])

with tab1:
    st.subheader("MCQ Assessment Architect")
    lo_in = st.text_input("Learning Outcome", "Human Respiration")
    c1, c2 = st.columns(2)
    q_num = c1.slider("Number of Questions", 5, 15, 10)
    tier_list = c2.multiselect("Difficulty Tiers", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation"])
    
    if st.button("🚀 Generate Structured MCQ Paper"):
        with st.spinner("Building your aligned assessment..."):
            test_json, aid = agent_assessment_creator(lo_in, q_num, tier_list)
            st.session_state.active_test = test_json['questions']
            st.session_state.active_aid = aid
            st.success(f"Assessment Generated! ID: {aid}")

    if 'active_test' in st.session_state:
        # Show Preview
        for i, q in enumerate(st.session_state.active_test[:3], 1): # Preview first 3
            st.write(f"**Q{i}: {q['question']}**")
            st.write(f"Options: {', '.join(q['options'])}")
        
        pdf_bytes = get_assessment_pdf(st.session_state.active_test, lo_in, st.session_state.active_aid)
        st.download_button("📥 Downl
