import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
from docx import Document
import io

# 1. SETUP & AUTHENTICATION
st.set_page_config(page_title="EduDiagnostic Elite 2026", layout="wide", page_icon="🎓")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key in Secrets! Add 'OPENAI_API_KEY' to your Streamlit settings.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. BRANDED PDF ENGINE (Logo Placeholder)
class ElitePDF(FPDF):
    def header(self):
        # Professional Header with Logo Space
        self.set_font('helvetica', 'B', 8)
        self.cell(45, 12, ' [ PLACE LOGO HERE ] ', 1, 0, 'C')
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(40, 70, 120)
        self.cell(0, 10, 'NEURO-DIAGNOSTIC ASSESSMENT', 0, 1, 'R')
        self.ln(10)

def get_pdf_bytes(text):
    pdf = ElitePDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    # Handle encoding for PDF generation
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    return bytes(pdf.output())

# 3. AGENT DEFINITIONS
def agent_architect(lo, count, tiers):
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. Tiers: {tiers}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a Senior Psychometrician. Output clean JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def agent_neuro_analyst(lo, student_data):
    data_json = student_data.to_json(orient='records')
    prompt = f"Perform a Deep Neuro-Diagnostic for LO: {lo}. PART 1: CLASS-WIDE REMEDIAL. PART 2: INDIVIDUAL PROFILES. Data: {data_json}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a Diagnostic Data Scientist."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. FORMATTING LAYER (The Beauty Layer)
def format_for_humans(json_data, lo_title):
    try:
        data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        output = f"DIAGNOSTIC TEST: {lo_title.upper()}\n"
        output += "NAME: __________________________    DATE: __________\n"
        output += "="*60 + "\n\n"
        for i, q in enumerate(data.get('questions', []), 1):
            output += f"Q{i} ({q.get('level')}): {q['question']}\n"
            for j, opt in enumerate(q['options']):
                output += f"   {chr(65+j)}) {opt}\n"
            output += "\n" + "."*40 + "\n\n"
        return output
    except:
        return "Error in formatting. Please retry generation."

# 5. USER INTERFACE (Unified Tabs)
st.title("🎓 EduDiagnostic Elite Multi-Agent System")
tab1, tab2 = st.tabs(["🏗️ Step 1: Create Assessment", "📊 Step 2: Deep Analysis"])

# --- TAB 1: CREATE ASSESSMENT ---
with tab1:
    st.subheader("Architect Your Professional Paper")
    colA, colB = st.columns(2)
    lo_input = colA.text_input("Learning Outcome", "Newton's Laws of Motion", key="lo_gen")
    q_num = colB.slider("Number of Questions", 5, 15, 7)
    tiers = st.multiselect("Select Difficulty Levels", 
                           ["Foundation", "Understanding", "Analytical", "Mastery"], 
                           ["Foundation", "Analytical"])

    # THE TRIGGER BUTTON (RESTORED)
    if st.button("🚀 Agent 1: Generate Branded Paper"):
        with st.spinner("AI Agent is crafting questions..."):
            raw_json = agent_architect(lo_input, q_num, tiers)
            st.session_state.clean_paper = format_for_humans(raw_json, lo_input)
            st.success("Paper Generated Successfully!")

    if 'clean_paper' in st.session_state:
        st.markdown("### 📝 Paper Preview")
        st.text_area("", st.session_state.clean_paper, height=300)
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download Branded PDF", data=get_pdf_bytes(st.session_state.clean_paper), file_name="Test.pdf")
        with c2:
            st.info("💡 PDF includes designated space for your School Logo.")

# --- TAB 2: DEEP ANALYSIS ---
with tab2:
    st.subheader("Deep Misconception Mapping")
    uploaded = st.file_uploader("Upload Student Marks (Excel)", type=["xlsx"])
    
    if uploaded:
        df = pd.read_excel(uploaded)
        if st.button("🧠 Agent 2: Run Analysis"):
            with st.spinner("Diagnosing misconception patterns..."):
                full_report = agent_neuro_analyst(lo_input, df)
                st.session_state.final_report = full_report
                st.markdown(full_report)

    if 'final_report' in st.session_state:
        st.download_button("📥 Download Multi-Colored Analysis (PDF)", 
                          data=get_pdf_bytes(st.session_state.final_report), 
                          file_name="Diagnostic_Report.pdf")
