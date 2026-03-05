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

# --- 2. ASSEMENTO BRANDED PDF ENGINE ---
class AssementoPDF(FPDF):
    def header(self):
        # Professional Branded Header
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 15, 'ASSEMENTO: NEURO-DIAGNOSTIC DOSSIER', 0, 1, 'C')
        self.ln(10)

def get_pdf_bytes(text):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    return bytes(pdf.output())

# --- 3. UTILITY: DYNAMIC EXCEL TEMPLATE ---
def create_excel_template(q_count, aid):
    columns = ["Student_Name"] + [f"Q{i+1}" for i in range(q_count)] + ["Score"]
    df_template = pd.DataFrame(columns=columns)
    # Add an instruction row with the ID
    df_template.loc[0] = [f"ID:{aid}"] + ["A/B/C/D"] * q_count + ["0-100"]
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False)
    return buffer.getvalue()

# --- 4. MULTI-AGENT DEFINITIONS ---
def agent_assessment_creator(lo, count, tiers):
    """Assemento Assessment Creator (Agent 1)"""
    aid = f"AID-{int(time.time())}"
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. ID: {aid}. Tiers: {tiers}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), aid

def agent_diagnostic_engine(lo, student_name, student_data, is_individual=False):
    """Assemento Diagnostic Engine (Agent 2)"""
    role_type = "Individual Student" if is_individual else "Class-wide"
    prompt = f"""
    ROLE: Assemento Diagnostic Engine. 
    LEVEL: {role_type} Analysis.
    LO: {lo}
    DATA: {student_data}
    FORMAT: Use the Recall-Relearn-Revise (R-R-R) protocol for insights.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 5. UI FLOW ---
st.title("🎯 Assemento Elite: Multi-Agent Intelligence")
tab1, tab2, tab3 = st.tabs(["🏗️ Creator", "📊 Class Engine", "👤 Individual Engine"])

# --- TAB 1: ASSEMENTO ASSESSMENT CREATOR ---
with tab1:
    st.subheader("Design Your Branded Assessment")
    c1, c2 = st.columns(2)
    lo_in = c1.text_input("Learning Outcome", "Photosynthesis")
    q_num = c2.slider("Questions", 5, 15, 8)
    
    if st.button("🚀 Generate Assessment & Template"):
        with st.spinner("Assemento is architecting..."):
            test_json, aid = agent_assessment_creator(lo_in, q_num, "Mixed")
            st.session_state.active_aid = aid
            st.session_state.q_count = q_num
            # Logic to format for preview...
            st.session_state.test_text = f"AID: {aid}\n\n" + str(test_json)
            st.success(f"Assessment Created! (ID: {aid})")

    if 'test_text' in st.session_state:
        st.download_button("📥 1. Download Test (PDF)", get_pdf_bytes(st.session_state.test_text), "Test.pdf")
        st.download_button("📥 2. Download Matching Template (Excel)", create_excel_template(st.session_state.q_count, st.session_state.active_aid), "Template.xlsx")

# --- TAB 2: ASSEMENTO CLASS ENGINE ---
with tab2:
    uploaded = st.file_uploader("Upload Completed Data Sheet", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        # Fingerprint Verification
        file_aid = str(df.iloc[0, 0])
        if st.session_state.get('active_aid') and st.session_state.active_aid not in file_aid:
            st.error("❌ Fingerprint Mismatch! This sheet does not belong to the active assessment.")
        else:
            st.success("✅ Fingerprint Verified.")
            st.plotly_chart(px.pie(df.iloc[1:], names='Score', title="Class Proficiency"), use_container_width=True)
            if st.button("🧠 Run Class Diagnostic"):
                report = agent_diagnostic_engine(lo_in, "Class", df.to_json())
                st.session_state.class_report = report
                st.markdown(report)

# --- TAB 3: ASSEMENTO INDIVIDUAL ENGINE ---
with tab3:
    if 'df' in locals() or 'class_report' in st.session_state:
        st.subheader("Student-Specific Neuro-Diagnostic")
        # Simplified for brevity: Logic to select student from df and call agent_diagnostic_engine with is_individual=True
        st.info("Select a student from the dropdown to generate their R-R-R Dossier.")
        # [Dropdown Logic Here]
