import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io
import time

# --- 1. BRANDING & STYLE ---
st.set_page_config(page_title="Assemento Elite 2026", layout="wide", page_icon="🎯")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key! Please add 'OPENAI_API_KEY' to Streamlit Secrets.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 2. ELITE PDF ENGINE (Branded & Structured) ---
class AssementoPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 15, 'ASSEMENTO: DEEP NEURO-DIAGNOSTIC REPORT', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def get_pdf_bytes(report_text, title):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(40, 70, 120)
    pdf.cell(0, 10, title, ln=True, align='L')
    pdf.line(10, 45, 200, 45)
    pdf.ln(10)
    
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    # Renders the deep report with clean line heights
    pdf.multi_cell(0, 7, txt=report_text.encode('latin-1', 'replace').decode('latin-1'))
    
    return bytes(pdf.output())

# --- 3. DEEP DIAGNOSTIC AGENT ---
def agent_diagnostic_engine(lo, student_data, scope="Class"):
    prompt = f"""
    ROLE: Elite Educational Neuro-Diagnostician.
    SCOPE: {scope} Analysis for LO: {lo}.
    DATA: {student_data}
    
    TASK: Generate an IN-DEPTH diagnostic report. Do not use generic summaries.
    
    STRUCTURE:
    1. COGNITIVE GAP ANALYSIS: Identify the exact mental models that are broken (e.g., 'Linear Reasoning Error').
    2. MISCONCEPTION HEATMAP: Which specific distractors are being confused and WHY?
    3. THE R-R-R REMEDIAL PLAN:
       - RECALL: Foundational facts the student must re-memorize.
       - RELEARN: Concepts the student must see through a 'Cognitive Conflict' (e.g., an experiment).
       - REVISE: A 3-day step-by-step action plan.
    4. PREDICTIVE RISK: What will this student struggle with in the NEXT chapter if this isn't fixed?
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 4. UI LOGIC ---
st.title("🎯 Assemento Elite: Multi-Agent Intelligence")
tab1, tab2 = st.tabs(["🏗️ Creator", "📊 Diagnostic Engine"])

with tab1:
    lo_in = st.text_input("Learning Outcome", "Human Respiratory System")
    # (Existing Creator Logic...)

with tab2:
    uploaded = st.file_uploader("Upload Student Marks (Excel)", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        st.write("### 📈 Visual Proficiency Overview")
        st.plotly_chart(px.pie(df, names='Score', title="Class Proficiency Breakdown", hole=0.4), use_container_width=True)
        
        # --- CLASS-WIDE REPORT ---
        if st.button("🧠 Generate In-Depth Class Report"):
            with st.spinner("Analyzing neural patterns..."):
                class_report = agent_diagnostic_engine(lo_in, df.to_json(), "Class")
                st.session_state.class_report = class_report
                st.markdown(class_report)
        
        if 'class_report' in st.session_state:
            st.download_button("📥 Download Class Report (PDF)", 
                              get_pdf_bytes(st.session_state.class_report, f"Class Analysis: {lo_in}"), 
                              "Class_Diagnostic.pdf")

        st.divider()
        
        # --- INDIVIDUAL REPORT DROPDOWN ---
        st.subheader("👤 Individual Student Dossier")
        student_list = df['Student_Name'].tolist()
        selected_student = st.selectbox("Select Student", student_list)
        
        if st.button(f"👤 Generate In-Depth Report for {selected_student}"):
            student_data = df[df['Student_Name'] == selected_student].to_json()
            with st.spinner(f"Diagnosing {selected_student}..."):
                indiv_report = agent_diagnostic_engine(lo_in, student_data, selected_student)
                st.session_state.indiv_report = indiv_report
                st.info(indiv_report)
        
        if 'indiv_report' in st.session_state:
            st.download_button("📥 Download Individual PDF", 
                              get_pdf_bytes(st.session_state.indiv_report, f"Report: {selected_student}"), 
                              f"Report_{selected_student}.pdf")
