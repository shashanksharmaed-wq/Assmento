import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io

# 1. SETUP & AUTHENTICATION
st.set_page_config(page_title="EduDiagnostic AI 2.0", layout="wide", page_icon="🎓")

# Retrieve API Key from Streamlit Cloud Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets! Check 'Settings' > 'Secrets'.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. PDF GENERATION ENGINE
class DiagnosticPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(40, 70, 120)
        self.cell(0, 10, 'Diagnostic Intelligence Report', 0, 1, 'C')
        self.ln(10)

def get_pdf_bytes(pdf_obj):
    """Ensures fpdf2 output is valid binary bytes for Streamlit."""
    return bytes(pdf_obj.output()) 

# 3. AGENT DEFINITIONS
def agent_creator(lo, count, diff_list):
    """Agent 1: Senior Psychometrician - Designs the assessment."""
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. Levels: {', '.join(diff_list)}. Output ONLY JSON."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a Senior Psychometrician."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def agent_analyst(lo, data):
    """Agent 2: Educational Data Analyst - Diagnoses class-wide gaps."""
    data_json = data.to_json(orient='records')
    prompt = f"Analyze these student responses for {lo}: {data_json}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an Educational Data Analyst."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. USER INTERFACE
st.title("🎓 EduDiagnostic Multi-Agent Engine")
st.markdown("---")

tab1, tab2 = st.tabs(["🏗️ Step 1: Create Assessment", "📊 Step 2: Analyze Results"])

with tab1:
    lo_input = st.text_input("Target Learning Outcome", key="lo_create")
    q_num = st.slider("Number of Questions", 5, 15, 10)
    tiers = st.multiselect("Difficulty Tiers", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation"])

    if st.button("🚀 Generate Test"):
        with st.spinner("Agent 1 is working..."):
            assessment = agent_creator(lo_input, q_num, tiers)
            st.session_state.current_test = assessment
            
            # PDF Generation
            pdf = DiagnosticPDF()
            pdf.add_page()
            pdf.set_font("helvetica", size=10)
            pdf.multi_cell(0, 8, txt=json.dumps(assessment, indent=2))
            
            st.download_button(
                label="📥 Download Print-Ready Test (PDF)",
                data=get_pdf_bytes(pdf),
                file_name=f"Test_{lo_input}.pdf",
                mime="application/pdf"
            )
            st.json(assessment)

with tab2:
    uploaded = st.file_uploader("Upload Student Marks (Excel)", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        if st.button("🧠 Run Analysis"):
            with st.spinner("Agent 2 is scanning patterns..."):
                report = agent_analyst(lo_input, df)
                st.markdown(report)
                
                r_pdf = DiagnosticPDF()
                r_pdf.add_page()
                r_pdf.set_font("helvetica", size=10)
                r_pdf.multi_cell(0, 8, txt=report)
                
                st.download_button("📥 Download Report (PDF)", data=get_pdf_bytes(r_pdf), file_name="Report.pdf")
