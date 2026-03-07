import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io
import time

# --- 1. SESSION STATE PERSISTENCE ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'active_test' not in st.session_state: st.session_state.active_test = None
if 'active_aid' not in st.session_state: st.session_state.active_aid = None
if 'df' not in st.session_state: st.session_state.df = None

# --- 2. 50-TEACHER LOGIN (T1-T50) ---
USER_DB = {f"T{i}": f"T{1233+i}" for i in range(1, 51)}

if not st.session_state.authenticated:
    st.title("🎯 Assemento Elite: Private Beta")
    with st.form("login"):
        u_id = st.text_input("Teacher ID (T1-T50)")
        p_wd = st.text_input("Password", type="password")
        if st.form_submit_button("Enter Engine"):
            if USER_DB.get(u_id) == p_wd:
                st.session_state.authenticated = True
                st.session_state.current_user = u_id
                st.rerun()
            else: st.error("Invalid Credentials.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 3. THE "STRUCTURED GRID" PDF ENGINE ---
class AssementoPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 15, f'ASSEMENTO ELITE - TEACHER: {st.session_state.current_user}', 0, 1, 'C')
        self.ln(10)

def generate_mcq_pdf(questions, title, aid):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14); pdf.cell(0, 10, title, ln=True, align='C')
    pdf.set_font("helvetica", "I", 9); pdf.cell(0, 5, f"ID: {aid}", ln=True, align='C'); pdf.ln(10)
    for i, q in enumerate(questions, 1):
        pdf.set_font("helvetica", "B", 11); pdf.multi_cell(0, 7, txt=f"Q{i}. {q['question']}")
        pdf.set_font("helvetica", "", 10); pdf.ln(2)
        for j, opt in enumerate(q['options']):
            pdf.set_x(20); pdf.cell(0, 7, txt=f"{chr(65+j)}) {opt}", ln=True)
        pdf.ln(5); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)
    return bytes(pdf.output())

# --- 4. MULTI-AGENT ENGINES (FIXED API CALLS) ---
def agent_creator(lo, count, tiers):
    aid = f"AID-{st.session_state.current_user}-{int(time.time())}"
    # OpenAI JSON mode requires the word "json" in the prompt
    prompt = f"Create a {count}-question MCQ for '{lo}' in JSON format. Tiers: {tiers}. Structure: 'questions': [{{'question', 'options':[], 'correct', 'misconception_map' }}]"
    
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Assemento Creator. You output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content), aid

def agent_diagnostic(lo, data, name=None):
    scope = f"Individual Report for {name}" if name else "Class-wide Analysis"
    prompt = f"In-depth {scope} for {lo}. Use Recall-Relearn-Revise (R-R-R) logic. DATA: {data}"
    
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Assemento Diagnostician. Provide high-depth remedial plans."},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

# --- 5. UI TABS ---
st.title(f"🚀 Assemento Elite: {st.session_state.current_user}")
tab1, tab2, tab3 = st.tabs(["🏗️ Assessment Creator", "📊 Class Engine", "👤 Individual Engine"])

with tab1:
    lo_in = st.text_input("Learning Outcome", "Human Physiology")
    q_num = st.slider("Test Length", 5, 15, 10)
    tiers = st.multiselect("Difficulty", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation"])
    if st.button("🚀 Generate Structured MCQ"):
        with st.spinner("AI is crafting the assessment..."):
            test_data, aid = agent_creator(lo_in, q_num, tiers)
            st.session_state.active_test, st.session_state.active_aid = test_data['questions'], aid
            st.success("Test Generated!")
    
    if st.session_state.active_test:
        st.download_button("📥 Download MCQ PDF", generate_mcq_pdf(st.session_state.active_test, lo_in, st.session_state.active_aid), "Assessment.pdf")

with tab2:
    up = st.file_uploader("Upload Data (Excel)", type=["xlsx"])
    if up:
        st.session_state.df = pd.read_excel(up)
        st.plotly_chart(px.pie(st.session_state.df, names='Score', title="Class Mastery", hole=0.4))
        if st.button("🧠 Run Class Diagnostic"):
            with st.spinner("Analyzing class misconceptions..."):
                report = agent_diagnostic(lo_in, st.session_state.df.to_json())
                st.session_state.class_report = report
                st.markdown(report)
        
        if 'class_report' in st.session_state:
            st.download_button("📥 Download Class Report PDF", generate_mcq_pdf([], "Class Diagnostic", "Summary"), "Class_Report.pdf")

with tab3:
    if st.session_state.df is not None:
        sel_s = st.selectbox("Select Student", st.session_state.df['Student_Name'].unique())
        if st.button(f"👤 Diagnose {sel_s}"):
            with st.spinner(f"Creating deep report for {sel_s}..."):
                s_data = st.session_state.df[st.session_state.df['Student_Name'] == sel_s].to_json()
                report = agent_diagnostic(lo_in, s_data, sel_s)
                st.session_state.last_report = report
                st.info(report)
        
        if 'last_report' in st.session_state:
            st.download_button("📥 Download Individual Report PDF", generate_mcq_pdf([], f"Individual Report: {sel_s}", "R-R-R"), f"Report_{sel_s}.pdf")
    else:
        st.warning("Please upload the Excel data sheet in the 'Class Engine' tab first.")
