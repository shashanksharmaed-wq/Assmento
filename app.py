import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io
import time

# --- 1. SECURE CREDENTIALS (T1-T50) ---
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

# --- 2. STRUCTURED PDF ENGINE ---
class AssementoPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 15, f'ASSEMENTO: TEACHER {st.session_state.current_user}', 0, 1, 'C')
        self.ln(10)

def get_mcq_pdf(questions, title, aid):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"TOPIC: {title}", ln=True, align='C')
    pdf.set_font("helvetica", "I", 9)
    pdf.cell(0, 5, f"Assessment ID: {aid}", ln=True, align='C')
    pdf.ln(10)
    
    for i, q in enumerate(questions, 1):
        # Clean Question Block
        pdf.set_font("helvetica", "B", 11)
        pdf.multi_cell(0, 7, txt=f"Q{i}. {q['question']}", border=0)
        pdf.ln(2)
        
        # Aligned Options Block
        pdf.set_font("helvetica", "", 10)
        for j, opt in enumerate(q['options']):
            pdf.set_x(20) # Clear indentation
            pdf.cell(0, 7, txt=f"{chr(65+j)}) {opt}", ln=True)
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
    return bytes(pdf.output())

# --- 3. AGENT FUNCTIONS (UNCHANGED) ---
def agent_assessment_creator(lo, count, tiers):
    aid = f"AID-{st.session_state.current_user}-{int(time.time())}"
    prompt = f"Create a {count}-question MCQ assessment for '{lo}'. Format: JSON with 'questions': [{{'question': '...', 'options': ['A', 'B', 'C', 'D']}}]. Difficulty: {tiers}."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), aid

def agent_diagnostic_engine(lo, student_data, name=None):
    scope = f"Individual Report for {name}" if name else "Class-wide Analysis"
    prompt = f"Perform {scope} for {lo}. Use Recall-Relearn-Revise (R-R-R) protocol. DATA: {student_data}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 4. UI FLOW ---
st.title(f"🚀 Assemento Elite: {st.session_state.current_user}")
tab1, tab2 = st.tabs(["🏗️ Creator", "📊 Diagnostic Engine"])

with tab1:
    st.subheader("Assessment Architect")
    lo_in = st.text_input("Learning Outcome", "Photosynthesis")
    q_num = st.slider("Questions", 5, 15, 8)
    tiers = st.multiselect("Tiers", ["Foundation", "Understanding", "Analytical"], ["Foundation"])
    
    if st.button("🚀 Run Assemento Creator"):
        with st.spinner("Crafting structured MCQ assessment..."):
            data, aid = agent_assessment_creator(lo_in, q_num, tiers)
            st.session_state.active_test = data['questions']
            st.session_state.active_aid = aid
            st.success("Test Generated!")

    if 'active_test' in st.session_state:
        st.download_button("📥 Download PDF (Structured MCQ)", 
                           get_mcq_pdf(st.session_state.active_test, lo_in, st.session_state.active_aid), 
                           "Assemento_Assessment.pdf")

with tab2:
    uploaded = st.file_uploader("Upload Data", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        if st.button("🧠 Run Deep Diagnostic"):
            st.markdown(agent_diagnostic_engine(lo_in, df.to_json()))
