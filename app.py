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

# --- 2. BRANDED PDF ENGINE (No Distortion Architecture) ---
class AssementoPDF(FPDF):
    def header(self):
        # Professional Branded Header with Logo Space
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 15, 'ASSEMENTO: NEURO-DIAGNOSTIC DOSSIER', 0, 1, 'C')
        self.ln(10)

def get_pdf_bytes(questions, title, aid):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title & Version Control
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(40, 70, 120)
    pdf.cell(0, 10, f"TOPIC: {title}", ln=True, align='C')
    pdf.set_font("helvetica", "I", 8)
    pdf.cell(0, 5, f"Assessment ID: {aid}", ln=True, align='C')
    
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, "Student Name: __________________________    Date: __________", ln=True)
    pdf.ln(5)

    for i, q in enumerate(questions, 1):
        # Question Block
        pdf.set_font("helvetica", "B", 11)
        pdf.multi_cell(0, 7, txt=f"Q{i}. {q['question']}")
        
        pdf.set_font("helvetica", "", 10)
        pdf.ln(2)
        
        # Options Block (Indented & Aligned)
        for j, opt in enumerate(q['options']):
            letter = chr(65 + j)
            pdf.set_x(20) # Standard Indentation
            pdf.cell(0, 7, txt=f"{letter}) {opt}", ln=True)
        
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Question Separator
        pdf.ln(5)

    return bytes(pdf.output())

# --- 3. UTILITY: DYNAMIC EXCEL TEMPLATE ---
def create_excel_template(q_count, aid):
    columns = ["Student_Name"] + [f"Q{i+1}" for i in range(q_count)] + ["Score"]
    df_template = pd.DataFrame(columns=columns)
    # The first row contains the metadata for Agent 2 to verify
    df_template.loc[0] = [f"ID:{aid}"] + ["Enter A/B/C/D"] * q_count + ["0-100"]
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False)
    return buffer.getvalue()

# --- 4. ASSEMENTO MULTI-AGENT DEFINITIONS ---
def agent_assessment_creator(lo, count, tiers):
    """Assemento Assessment Creator (Agent 1)"""
    aid = f"AID-{int(time.time())}"
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. ID: {aid}. Tiers: {tiers}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator. Ensure pedagogical accuracy."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), aid

def agent_diagnostic_engine(lo, student_data, student_name=None):
    """Assemento Diagnostic Engine (Agent 2)"""
    scope = f"Individual Report for {student_name}" if student_name else "Class-wide Analysis"
    prompt = f"Perform a {scope} for LO: {lo}. Identify misconceptions and provide Recall-Relearn-Revise (R-R-R) plans. Data: {student_data}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 5. USER INTERFACE FLOW ---
st.title("🎯 Assemento Elite: Multi-Agent Intelligence")
tab1, tab2, tab3 = st.tabs(["🏗️ Assemento Assessment Creator", "📊 Class Engine", "👤 Individual Engine"])

# --- TAB 1: CREATOR ---
with tab1:
    st.subheader("Design Your Branded Assessment")
    col1, col2 = st.columns(2)
    lo_in = col1.text_input("Learning Outcome", "Human Circulatory System")
    q_num = col2.slider("Test Length", 5, 15, 8)
    tiers = st.multiselect("Difficulty Levels", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation", "Analytical"])
    
    if st.button("🚀 Run Assessment Creator"):
        with st.spinner("Assemento is architecting your aligned paper..."):
            test_json, aid = agent_assessment_creator(lo_in, q_num, tiers)
            st.session_state.active_aid = aid
            st.session_state.q_count = q_num
            st.session_state.raw_questions = test_json.get('questions', [])
            st.session_state.pdf_paper = get_pdf_bytes(st.session_state.raw_questions, lo_in, aid)
            st.success(f"Assessment Created Successfully! (ID: {aid})")

    if 'pdf_paper' in st.session_state:
        c_a, c_b = st.columns(2)
        with c_a:
            st.download_button("📥 1. Download Aligned PDF", st.session_state.pdf_paper, "Assemento_Test.pdf")
        with c_b:
            template_bytes = create_excel_template(st.session_state.q_count, st.session_state.active_aid)
            st.download_button("📥 2. Download Data Sheet (Excel)", template_bytes, "Assemento_Template.xlsx")

# --- TAB 2: CLASS ENGINE ---
with tab2:
    st.subheader("Class-Wide Diagnostic Intelligence")
    uploaded = st.file_uploader("Upload Completed Data Sheet", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        file_aid = str(df.iloc[0, 0])
        
        # Security Fingerprint Check
        if st.session_state.get('active_aid') and st.session_state.active_aid not in file_aid:
            st.error("❌ Fingerprint Mismatch! This data sheet does not match the active Assessment ID.")
        else:
            st.success("✅ Fingerprint Verified.")
            # Visualize Proficiency
            if 'Score' in df.columns:
                fig = px.pie(df.iloc[1:], names='Score', title="Class Proficiency Breakdown", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🧠 Run Class Diagnostic Engine"):
                with st.spinner("Analyzing class-wide neural patterns..."):
                    class_report = agent_diagnostic_engine(lo_in, df.to_json())
                    st.session_state.class_report = class_report
                    st.markdown(class_report)

# --- TAB 3: INDIVIDUAL ENGINE ---
with tab3:
    if uploaded is not None:
        st.subheader("Student-Specific R-R-R Dossier")
        student_list = df.iloc[1:]['Student_Name'].tolist()
        selected_student = st.selectbox("Select Student to Diagnose", student_list)
        
        if st.button(f"👤 Generate Report for {selected_student}"):
            student_data = df[df['Student_Name'] == selected_student].to_json()
            with st.spinner(f"Diagnosing {selected_student}..."):
                indiv_report = agent_diagnostic_engine(lo_in, student_data, selected_student)
                st.info(indiv_report)
                st.download_button("📥 Download Individual PDF", get_pdf_bytes([], f"Report: {selected_student}", "Individual"), f"Report_{selected_student}.pdf")
    else:
        st.warning("Please upload the data sheet in Tab 2 first.")
