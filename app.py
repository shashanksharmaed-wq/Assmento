import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io
import time

# --- 1. SECURE CREDENTIALS GENERATION (T1 to T50) ---
# This dictionary now contains all 50 pairs exactly as you requested.
USER_DB = {f"T{i}": f"T{1233+i}" for i in range(1, 51)}

# --- 2. LOGIN GATEKEEPER ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🎯 Assemento Elite: Private Beta")
    st.info("Authorized Access Only: Enter your Teacher ID (T1-T50)")
    
    with st.form("login_form"):
        u_id = st.text_input("Teacher ID (e.g., T1)")
        p_wd = st.text_input("Password", type="password")
        if st.form_submit_button("Enter Engine"):
            if USER_DB.get(u_id) == p_wd:
                st.session_state.authenticated = True
                st.session_state.current_user = u_id
                st.rerun()
            else:
                st.error("Invalid ID or Password. Please check your credentials.")
    st.stop()

# --- 3. CORE ENGINE CONFIGURATION (Do Not Change) ---
st.set_page_config(page_title="Assemento Elite 2026", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

class AssementoPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 15, f'ASSEMENTO ELITE - TEACHER: {st.session_state.current_user}', 0, 1, 'C')
        self.ln(10)

def get_pdf_bytes(questions, title, aid):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"TOPIC: {title} (ID: {aid})", ln=True, align='C')
    pdf.ln(5)
    for i, q in enumerate(questions, 1):
        pdf.set_font("helvetica", "B", 11)
        pdf.multi_cell(0, 7, txt=f"Q{i}. {q['question']}")
        pdf.set_font("helvetica", "", 10)
        for j, opt in enumerate(q['options']):
            pdf.set_x(20)
            pdf.cell(0, 7, txt=f"{chr(65+j)}) {opt}", ln=True)
        pdf.ln(5)
    return bytes(pdf.output())

# --- 4. MULTI-AGENT FUNCTIONS (UNCHANGED) ---
def agent_assessment_creator(lo, count):
    aid = f"AID-{st.session_state.current_user}-{int(time.time())}"
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. ID: {aid}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), aid

def agent_diagnostic_engine(lo, student_data, name=None):
    scope = f"Individual Report for {name}" if name else "Class-wide Analysis"
    prompt = f"Perform a {scope} for LO: {lo}. Identify misconceptions and provide Recall-Relearn-Revise (R-R-R) plans. Data: {student_data}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 5. UI FLOW ---
st.sidebar.title(f"👤 {st.session_state.current_user}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

tab1, tab2, tab3 = st.tabs(["🏗️ Creator", "📊 Class Engine", "👤 Individual Engine"])

with tab1:
    st.subheader("Create Assessment")
    lo_in = st.text_input("Learning Outcome", "Photosynthesis")
    q_num = st.slider("Questions", 5, 15, 8)
    if st.button("🚀 Run Assemento Creator"):
        test_json, aid = agent_assessment_creator(lo_in, q_num)
        st.session_state.active_aid = aid
        st.session_state.q_list = test_json['questions']
        st.session_state.pdf_file = get_pdf_bytes(st.session_state.q_list, lo_in, aid)
        st.success(f"Assessment Created! ID: {aid}")
    if 'pdf_file' in st.session_state:
        st.download_button("📥 Download PDF", st.session_state.pdf_file, "Test.pdf")

with tab2:
    st.subheader("Class Analysis")
    uploaded = st.file_uploader("Upload Marks Excel", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        st.plotly_chart(px.pie(df, names='Score', title="Class Proficiency Breakdown", hole=0.4))
        if st.button("🧠 Run Class Diagnostic"):
            st.markdown(agent_diagnostic_engine(lo_in, df.to_json()))

with tab3:
    if 'df' in locals():
        st.subheader("Individual R-R-R Report")
        s_list = df['Student_Name'].tolist()
        sel_s = st.selectbox("Select Student", s_list)
        if st.button(f"👤 Diagnose {sel_s}"):
            s_data = df[df['Student_Name'] == sel_s].to_json()
            st.info(agent_diagnostic_engine(lo_in, s_data, sel_s))
