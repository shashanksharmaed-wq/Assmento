import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import plotly.express as px
import io

# 1. SETUP & BRANDING
st.set_page_config(page_title="Assemento Elite 2026", layout="wide", page_icon="🎯")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing API Key! Please add 'OPENAI_API_KEY' to Streamlit Secrets.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. BRANDED PDF ENGINE
class AssementoPDF(FPDF):
    def header(self):
        self.set_fill_color(40, 70, 120)
        self.rect(0, 0, 210, 20, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 14)
        self.cell(0, 10, 'ASSEMENTO DIAGNOSTIC INTELLIGENCE', 0, 1, 'C')
        self.ln(10)

def get_pdf_bytes(text):
    pdf = AssementoPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    return bytes(pdf.output())

# 3. ASSEMENTO AGENTS
def agent_assessment_creator(lo, count, tiers):
    """Agent 1: Assemento Assessment Creator"""
    prompt = f"Create a {count}-question diagnostic for LO: {lo}. Tiers: {tiers}. Output ONLY JSON: 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Assessment Creator. Output clean JSON only."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def agent_diagnostic_engine(lo, student_data):
    """Agent 2: Assemento Diagnostic Engine"""
    data_json = student_data.to_json(orient='records')
    prompt = f"Perform a Deep Neuro-Diagnostic for LO: {lo}. Identify Class-wide Thinking Traps and Individual Remediation. Data: {data_json}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are the Assemento Diagnostic Engine."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 4. FORMATTING LAYER
def format_for_humans(json_data, lo_title):
    try:
        data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        output = f"DIAGNOSTIC TEST: {lo_title.upper()}\n"
        output += "NAME: __________________________    DATE: __________\n"
        output += "="*60 + "\n\n"
        for i, q in enumerate(data.get('questions', []), 1):
            output += f"Q{i} ({q.get('level')}): {q['question']}\n"
            for j, opt in enumerate(q['options']):
                output += f"   {chr(65+j)}) {opt}\n"
            output += "\n" + "."*40 + "\n\n"
        return output
    except:
        return "Error in formatting. Please retry."

# 5. USER INTERFACE (TABS)
st.title("🎯 Assemento Elite: Multi-Agent Intelligence")
tab1, tab2 = st.tabs(["🏗️ Assemento Assessment Creator", "📊 Assemento Diagnostic Engine"])

# --- TAB 1: CREATOR ---
with tab1:
    st.subheader("Design Your Branded Paper")
    col1, col2 = st.columns(2)
    lo_input = col1.text_input("Learning Outcome", "Energy Conservation", key="lo_cre")
    q_num = col2.slider("Test Length", 5, 15, 8)
    tiers = st.multiselect("Difficulty Levels", ["Foundation", "Understanding", "Analytical", "Mastery"], ["Foundation"])

    if st.button("🚀 Run Assessment Creator"):
        with st.spinner("Assemento is architecting questions..."):
            raw_json = agent_assessment_creator(lo_input, q_num, tiers)
            st.session_state.clean_paper = format_for_humans(raw_json, lo_input)
            st.success("Paper Generated Successfully!")

    if 'clean_paper' in st.session_state:
        st.text_area("Final Paper Preview", st.session_state.clean_paper, height=300)
        st.download_button("📥 Download Branded PDF", data=get_pdf_bytes(st.session_state.clean_paper), file_name="Assemento_Test.pdf")

# --- TAB 2: DIAGNOSTIC ENGINE ---
with tab2:
    st.subheader("Visual Misconception Mapping")
    uploaded = st.file_uploader("Upload Student Marks (Excel)", type=["xlsx"])
    
    if uploaded:
        df = pd.read_excel(uploaded)
        
        # --- GRAPHICAL REPORTS ---
        st.write("### 📈 Visual Overview")
        g_col1, g_col2 = st.columns(2)
        
        # Graph 1: Score Distribution (Bar Chart)
        if 'Score' in df.columns:
            fig1 = px.histogram(df, x="Score", nbins=10, title="Class Score Distribution", color_discrete_sequence=['#284678'])
            g_col1.plotly_chart(fig1, use_container_width=True)
            
            # Graph 2: Performance Levels (Pie Chart)
            performance_counts = df['Score'].apply(lambda x: 'High' if x > 80 else ('Medium' if x > 50 else 'Low')).value_counts()
            fig2 = px.pie(names=performance_counts.index, values=performance_counts.values, title="Proficiency Breakdown", hole=0.4)
            g_col2.plotly_chart(fig2, use_container_width=True)

        if st.button("🧠 Run Diagnostic Engine"):
            with st.spinner("Assemento is identifying neural logic patterns..."):
                full_report = agent_diagnostic_engine(lo_input, df)
                st.session_state.final_report = full_report
                st.markdown(full_report)

    if 'final_report' in st.session_state:
        st.download_button("📥 Download Full Analysis (PDF)", data=get_pdf_bytes(st.session_state.final_report), file_name="Assemento_Report.pdf")
