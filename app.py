import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import io

# 1. SETUP & AUTHENTICATION
st.set_page_config(page_title="EduDiagnostic AI 2.0", layout="wide", page_icon="🎓")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key in Streamlit Secrets! Check settings.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. PDF GENERATOR
class DiagnosticPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Diagnostic Intelligence Dossier', 0, 1, 'C')
        self.ln(10)

# 3. AGENT DEFINITIONS (Modular logic)
def agent_psychometrician(lo, count, diff):
    """Generates the assessment with built-in misconception traps."""
    prompt = f"""
    ROLE: Senior Psychometrician.
    TASK: Create a {count}-question diagnostic for LO: {lo}.
    LEVELS: {diff}.
    OUTPUT: JSON ONLY with structure: 
    {{ "questions": [{{ "id":1, "level":"", "question":"", "options":[], "correct":index, "misconception_map": {{ "index": "thinking error" }} }}] }}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def agent_analyst(lo, excel_data):
    """Analyzes raw Excel data to identify patterns and remedial paths."""
    prompt = f"""
    ROLE: Educational Data Analyst.
    INPUT: Student response data for LO: {lo}.
    TASK: Identify top 3 class-wide misconceptions and create a 3-step remedial plan.
    DATA: {excel_data.to_json()}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Analyze and output pedagogical steps."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. APP INTERFACE (TABS)
st.title("🎓 EduDiagnostic Multi-Agent Engine")
tab_create, tab_analyze = st.tabs(["🏗️ Agent 1: Create Test", "📊 Agent 2: Analyze Results"])

# --- TAB: CREATE ASSESSMENT ---
with tab_create:
    lo_input = st.text_input("Enter Learning Outcome", "e.g., Newton's Second Law", key="lo_create")
    st.write("### Define Test Strategy")
    c1, c2 = st.columns(2)
    q_count = c1.slider("Question Count", 5, 20, 10)
    levels = c2.multiselect("Target Levels", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation"])
    
    if st.button("🚀 Agent 1: Architect Assessment"):
        with st.spinner("Psychometrician Agent is drafting questions..."):
            st.session_state.assessment = agent_psychometrician(lo_input, q_count, levels)
            st.success("Assessment Generated!")
            
            # Display Questions
            for q in st.session_state.assessment['questions']:
                with st.expander(f"Q{q['id']} - {q['level']}"):
                    st.write(q['question'])
                    st.write(q['options'])
            
            # PDF Generation
            pdf = DiagnosticPDF()
            pdf.add_page()
            pdf.set_font("helvetica", size=11)
            pdf.multi_cell(0, 10, txt=json.dumps(st.session_state.assessment, indent=2))
            st.download_button("📥 Download Print-Ready Test", data=pdf.output(), file_name="Test.pdf")

# --- TAB: ANALYZE RESULTS ---
with tab_analyze:
    st.subheader("Process Paper-Test Data")
    uploaded = st.file_uploader("Upload Student Marks (Excel)", type=["xlsx"])
    
    if uploaded:
        df = pd.read_excel(uploaded)
        if st.button("🧠 Agent 2: Diagnose Gaps"):
            with st.spinner("Analyst Agent is scanning student patterns..."):
                report = agent_analyst(lo_input, df)
                st.markdown(report)
                
                # Report PDF
                r_pdf = DiagnosticPDF()
                r_pdf.add_page()
                r_pdf.set_font("helvetica", size=10)
                r_pdf.multi_cell(0, 8, txt=report)
                st.download_button("📥 Download Remedial Plan (PDF)", data=r_pdf.output(), file_name="Report.pdf")
