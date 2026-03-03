import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
from docx import Document
import io

# 1. INITIALIZATION & LOGO UPLOAD
st.set_page_config(page_title="EduDiagnostic Elite 2026", layout="wide", page_icon="🎓")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key! Please add it to Streamlit Secrets.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. ENHANCED PDF ENGINE (With Logo Space & Colors)
class ElitePDF(FPDF):
    def header(self):
        # Placeholder for School Logo (Top Left)
        self.set_font('helvetica', 'B', 8)
        self.cell(40, 10, '[ INSERT SCHOOL LOGO HERE ]', 1, 0, 'C')
        
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(40, 70, 120) # Professional Blue
        self.cell(0, 10, 'NEURO-DIAGNOSTIC REPORT', 0, 1, 'R')
        self.ln(10)

def get_pdf_bytes(text):
    pdf = ElitePDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    return bytes(pdf.output())

# 3. AGENT DEFINITIONS
def agent_neuro_analyst(lo, student_data):
    """Deep analysis for both Class-wise and Individual remediation."""
    data_json = student_data.to_json(orient='records')
    prompt = f"""
    ROLE: Elite Educational Neuro-Diagnostician.
    LO: {lo}
    TASK: Generate a 2-Part Deep Diagnostic.
    PART 1: CLASS-WIDE REMEDIAL (Identify 'Thinking Traps' affecting >30% of students).
    PART 2: INDIVIDUAL STUDENT PROFILES (Specific gaps for every student).
    Format the output with clear headers and bullet points.
    DATA: {data_json}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a Diagnostic Data Scientist."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. USER INTERFACE
st.title("🎓 EduDiagnostic Elite Multi-Agent System")
st.markdown("---")

tab1, tab2 = st.tabs(["🏗️ Step 1: Create Test", "📊 Step 2: Deep Analysis"])

with tab1:
    lo_input = st.text_input("Learning Outcome", "Human Circulatory System")
    # (Previous Create Test logic remains here...)
    st.info("Test Creation Agent is active. Generate your paper first.")

with tab2:
    st.subheader("Deep Misconception Mapping")
    uploaded = st.file_uploader("Upload Student Marks (Excel)", type=["xlsx"])
    
    if uploaded:
        df = pd.read_excel(uploaded)
        
        # VISUAL APPEAL: Color-coded Data Preview
        st.write("### 📥 Raw Data Ingested")
        st.dataframe(df.style.highlight_null(color='lightcoral')) 

        if st.button("🧠 Run Multi-Agent Diagnostic"):
            with st.spinner("Analyzing neural logic patterns..."):
                report = agent_neuro_analyst(lo_input, df)
                st.session_state.elite_report = report
                
                # Visual Sections
                st.success("Analysis Complete!")
                st.markdown("### 🗺️ Class-Wide Remedial Path")
                st.info(report.split("PART 2")[0]) # Shows Class-wide first
                
                st.markdown("### 👤 Individual Student Insights")
                st.warning("PART 2" + report.split("PART 2")[-1]) # Shows Individual

    if 'elite_report' in st.session_state:
        st.divider()
        st.download_button("📥 Download Branded PDF Report", 
                          data=get_pdf_bytes(st.session_state.elite_report), 
                          file_name="Deep_Diagnostic_Report.pdf")
