import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io
import time

# --- 1. BRANDING & LOGIN (T1 to T50) ---
st.set_page_config(page_title="Assemento Elite 2026", layout="wide", page_icon="🎯")
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

# --- 2. ELITE PDF ENGINE ---
class AssementoPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 25, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 15, f'ASSEMENTO: {st.session_state.get("current_user")} - NEURO-DIAGNOSTIC', 0, 1, 'C')
        self.ln(10)

def get_pdf_bytes(text, title):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 7, txt=text.encode('latin-1', 'replace').decode('latin-1'))
    return bytes(pdf.output())

# --- 3. MULTI-AGENT FUNCTIONS ---
def agent_assessment_creator(lo, count, tiers):
    aid = f"AID-{st.session_state.current_user}-{int(time.time())}"
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. Tiers: {tiers}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content), aid

def agent_diagnostic_engine(lo, student_data, scope="Class"):
    prompt = f"Perform an IN-DEPTH diagnostic for LO: {lo}. Scope: {scope}. Use Recall-Relearn-Revise. Data: {student_data}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 4. MAIN INTERFACE ---
st.title(f"🚀 Assemento Elite: {st.session_state.current_user}")
tab1, tab2 = st.tabs(["🏗️ Creator", "📊 Diagnostic Engine"])

# RESTORED TAB 1 CONTROLS
with tab1:
    st.subheader("Assessment Architect")
    lo_in = st.text_input("Learning Outcome", "Newton's Laws")
    c1, c2 = st.columns(2)
    q_num = c1.slider("Number of Questions", 5, 15, 8)
    tiers = c2.multiselect("Difficulty Tiers", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation", "Analytical"])
    
    # RESTORED BUTTON
    if st.button("🚀 Run Assemento Creator"):
        with st.spinner("Generating tiered assessment..."):
            test_json, aid = agent_assessment_creator(lo_in, q_num, tiers)
            st.session_state.active_test = test_json
            st.session_state.active_aid = aid
            st.success(f"Assessment Created! ID: {aid}")

    if 'active_test' in st.session_state:
        st.download_button("📥 Download Test (PDF)", 
                          get_pdf_bytes(str(st.session_state.active_test), f"Test: {lo_in}"), 
                          "Test.pdf")

# TAB 2 ENGINE
with tab2:
    uploaded = st.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded:
        df = pd.read_excel(uploaded)
        st.plotly_chart(px.pie(df, names='Score', title="Class Proficiency", hole=0.4))
        
        if st.button("🧠 Run Deep Diagnostic"):
            report = agent_diagnostic_engine(lo_in, df.to_json())
            st.session_state.last_report = report
            st.markdown(report)
        
        if 'last_report' in st.session_state:
            st.download_button("📥 Download Diagnostic PDF", 
                              get_pdf_bytes(st.session_state.last_report, "Neuro-Diagnostic Report"), 
                              "Report.pdf")
