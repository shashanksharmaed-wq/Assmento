import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
from docx import Document
import io

# 1. SETUP & AUTHENTICATION
st.set_page_config(page_title="EduDiagnostic Pro 2026", layout="wide", page_icon="🎓")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets! Check 'Settings' > 'Secrets'.")
    st.stop()

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. UTILITY: THE BEAUTY LAYER (Formatting for Humans)
def format_for_print(json_data, lo_title):
    try:
        data = json_data if isinstance(json_data, dict) else json.loads(json_data)
        output = f"DIAGNOSTIC ASSESSMENT: {lo_title.upper()}\n"
        output += "NAME: __________________________    DATE: __________\n"
        output += "="*60 + "\n\n"
        
        for i, q in enumerate(data.get('questions', []), 1):
            output += f"Q{i} [{q.get('level', 'Core')}]: {q['question']}\n\n"
            for j, opt in enumerate(q.get('options', [])):
                output += f"   {chr(65+j)}) {opt}\n"
            output += "\n" + "-"*30 + "\n\n"
        return output
    except Exception as e:
        return f"Formatting Error: {str(e)}"

# 3. DOWNLOAD UTILITIES (PDF & WORD)
def get_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    # Using 'replace' to handle special characters safely in latin-1
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=safe_text)
    return bytes(pdf.output())

def get_docx_bytes(text):
    doc = Document()
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# 4. AGENT DEFINITIONS
def agent_architect(lo, count, tiers):
    """Creates the test questions based on LO and Difficulty Levels."""
    prompt = f"""
    Create a {count}-question diagnostic for LO: {lo}. 
    Include these Difficulty Tiers: {', '.join(tiers)}.
    Output ONLY JSON with 'questions': [{{id, level, question, options:[], correct, misconception_map:{{index:reason}} }}]
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a Senior Psychometrician. Output clean JSON."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def agent_analyst(lo, student_data):
    """Analyzes student Excel results to identify misconception patterns."""
    data_json = student_data.to_json(orient='records')
    prompt = f"""
    Analyze these student results for the Learning Outcome: {lo}.
    1. Identify the top 3 class-wide misconceptions.
    2. Provide a 3-step remedial plan for the teacher.
    3. Generate a summary of individual student gaps.
    Data: {data_json}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an Educational Data Analyst."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 5. USER INTERFACE (TABS)
st.title("🎓 EduDiagnostic: Create & Analyze")
tab1, tab2 = st.tabs(["🏗️ Step 1: Create Test", "📊 Step 2: Analyze Results"])

# --- TAB 1: CREATE ASSESSMENT ---
with tab1:
    lo_input = st.text_input("Enter Learning Outcome (LO)", "Human Circulatory System", key="lo1")
    c1, c2 = st.columns(2)
    q_num = c1.slider("Test Length (Questions)", 5, 20, 10)
    tiers = c2.multiselect("Difficulty Tiers", 
                           ["Foundation", "Understanding", "Analytical", "Mastery"], 
                           ["Foundation", "Understanding"])
    
    if st.button("🚀 Agent 1: Generate Printable Test"):
        with st.spinner("Architecting questions with specific difficulty levels..."):
            raw_test = agent_architect(lo_input, q_num, tiers)
            st.session_state.clean_test = format_for_print(raw_test, lo_input)
            st.success("Test Paper Generated!")

    if 'clean_test' in st.session_state:
        st.text_area("Live Preview (Brackets Removed)", st.session_state.clean_test, height=300)
        c3, c4 = st.columns(2)
        with c3:
            st.download_button("📥 PDF Test", data=get_pdf_bytes(st.session_state.clean_test), file_name="Test.pdf")
        with c4:
            st.download_button("📥 Word Test", data=get_docx_bytes(st.session_state.clean_test), file_name="Test.docx")

# --- TAB 2: ANALYZE RESULTS ---
with tab2:
    st.subheader("Process Pen-and-Paper Results (Excel)")
    uploaded = st.file_uploader("Upload Excel with Student Choices", type=["xlsx"])
    
    if uploaded:
        df = pd.read_excel(uploaded)
        st.dataframe(df.head()) # Preview the data
        if st.button("🧠 Agent 2: Run Misconception Analysis"):
            with st.spinner("Analyst is identifying misconception patterns..."):
                report = agent_analyst(lo_input, df)
                st.session_state.final_report = report
                st.markdown(report)

    if 'final_report' in st.session_state:
        st.download_button("📥 Download Report (PDF)", 
                          data=get_pdf_bytes(st.session_state.final_report), 
                          file_name="Diagnostic_Report.pdf")
