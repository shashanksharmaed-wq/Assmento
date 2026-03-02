import streamlit as st
from fpdf import FPDF

def generate_assessment_pdf(assessment_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # Add your assessment text
    pdf.multi_cell(0, 10, txt=assessment_content)
    
    # NANO-FIX: Convert to bytes explicitly for Streamlit
    # pdf.output() returns bytes in newer fpdf2 versions; 
    # use .encode('latin-1') if using older versions or dest='S'
    return bytes(pdf.output())

# Example Usage in your app
assessment_text = "Q1: What is Newton's Second Law? ..."
pdf_bytes = generate_assessment_pdf(assessment_text)

st.download_button(
    label="📥 Download Assessment PDF",
    data=pdf_bytes,
    file_name="assessment.pdf",
    mime="application/pdf"
)
