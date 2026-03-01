import streamlit as st
import pandas as pd
import openai
import json
from fpdf import FPDF
import io

# 1. SETUP & AUTHENTICATION
st.set_page_config(page_title="EduDiagnostic Pro 2026", layout="wide", page_icon="📝")

# Accessing the secret key from Streamlit's cloud settings
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("Please configure your OPENAI_API_KEY in the Streamlit Cloud Secrets.")
    st.stop()

# 2. PDF GENERATION
class DiagnosticPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Misconception Diagnostic Report', 0, 1, 'C')
        self.ln(10)

def create_pdf_report(content):
    pdf = DiagnosticPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 10, txt=content)
    return pdf.output()

# 3. SAMPLE EXCEL GENERATOR
def get_sample_excel():
    buffer = io.BytesIO()
    sample_data = {
        "Student_Name": ["Rahul S.", "Priya K.", "Aman V."],
        "Q1_Level1": ["A", "B", "A"],
        "Q2_Level2": ["C", "C", "D"],
        "Q3_Level3": ["B", "A", "B"]
    }
    df_sample = pd.DataFrame(sample_data)
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_sample.to_excel(writer, index=False)
    return buffer

# 4. MAIN INTERFACE
st.title("🎯 Diagnostic Assessment Engine")
st.write("Convert pen-and-paper test results into deep pedagogical insights.")

with st.sidebar:
    st.header("1. Setup Template")
    st.download_button(
        label="📥 Download Sample Excel Template",
        data=get_sample_excel(),
        file_name="student_test_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.divider()
    lo = st.text_input("Learning Outcome", "Newton's Second Law")

# 5. DATA UPLOAD & ANALYSIS
uploaded_file = st.file_uploader("Upload your filled Excel here", type=["xlsx"])

if uploaded_file and lo:
    df = pd.read_excel(uploaded_file)
    st.success("Data uploaded! Ready for EI Analysis.")
    
    if st.button("Generate Diagnostic Report"):
        with st.spinner("Analyzing mental models..."):
            # Prepare data for AI
            data_str = df.to_json(orient='records')
            
            prompt = f"Analyze these student responses for the Learning Outcome: {lo}. " \
                     f"Identify class-wide misconceptions and provide a remedial plan. Data: {data_str}"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            report_text = response.choices[0].message.content
            st.markdown(report_text)
            
            # PDF Export
            pdf_bytes = create_pdf_report(report_text)
            st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="Diagnostic_Report.pdf")
