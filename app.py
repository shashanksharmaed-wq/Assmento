import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# --- 1. SYSTEM ORCHESTRATION ---
st.set_page_config(page_title="RemediAI Ultra | Multi-Agent Engine", layout="wide")

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("🔑 OpenAI API Key Missing!")
    st.stop()

# Shared Memory (The Vault)
if 'vault' not in st.session_state: st.session_state['vault'] = {}

# --- AGENT A: THE PSYCHOMETRICIAN ---
# Specialized in distractor engineering and scenario creation
def agent_psychometrician(topic, grade, num_q, diff):
    prompt = f"""
    Role: Senior Psychometrician at EI ASSET.
    Task: Create {num_q} 'Deep Diagnostic' MCQs for Grade {grade} on {topic}.
    Difficulty: {diff}/12.
    
    CONSTRAINTS:
    - Use 'Scenario-Based' questions (e.g., 'If X happened, what would be the result?').
    - Every wrong option (B, C, D) must represent a 'Smart Distractor' (a common misconception).
    - NO literal or knowledge-based questions.
    - NO diagrams.
    
    OUTPUT FORMAT: JSON ONLY
    {{ "questions": [ {{ "id": 1, "q": "...", "options": {{"A":"","B":"","C":"","D":""}}, "correct": "A", "mappings": {{"B":"Misconception Description", "C":"Misconception Description", "D":"Logic Error"}}, "remedy": "Pedagogical intervention." }} ] }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master of educational diagnostic design."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT B: THE DOCUMENT ARCHITECT ---
# Specialized in multicolored, branded, print-ready PDF design
def agent_document_architect(metadata, info, school_name):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    
    # Branded Header (Deep Blue)
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, h - 90, w, 90, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(w/2, h - 40, school_name.upper())
    p.setFont("Helvetica", 10)
    p.drawCentredString(w/2, h - 60, f"DIAGNOSTIC ASSESSMENT | {info['aid']}")
    
    # Metadata Bar (Grey)
    p.setLineWidth(1)
    p.setStrokeColor(colors.lightgrey)
    p.setFillColor(colors.HexColor("#f1f5f9"))
    p.rect(40, h - 120, w - 80, 25, fill=1, stroke=1)
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 9)
    p.drawString(50, h - 113, f"GRADE: {info['grade']}  |  TOPIC: {info['topic']}")
    p.drawRightString(w - 50, h - 113, f"TIME: {info['time']} MINS")

    # Question Layout
    y = h - 160
    for q in metadata.get('questions', []):
        if y < 150: p.showPage(); y = h - 50
        p.setFont("Helvetica-Bold", 11)
        p.setFillColor(colors.HexColor("#1e3a8a"))
        p.drawString(50, y, f"Q{q['id']}.")
        p.setFillColor(colors.black)
        
        # Simple Wrapping
        text = q.get('q') or q.get('question')
        p.drawString(75, y, text[:80])
        if len(text) > 80:
            y -= 15
            p.drawString(75, y, text[80:])
            
        y -= 25
        p.setFont("Helvetica", 10)
        opts = q.get('options', {})
        for lbl in ["A", "B", "C", "D"]:
            p.drawString(85, y, f"{lbl}) {opts.get(lbl)}")
            y -= 18
        y -= 20

    p.save()
    buffer.seek(0)
    return buffer

# --- AGENT C: DATA DIAGNOSTIC ENGINE ---
# Handles Excel templates and mapping results back to misconceptions
def agent_data_engine(aid, uploaded_file):
    if aid not in st.session_state['vault']:
        return None, "Assessment ID not found."
    
    results_df = pd.read_excel(uploaded_file)
    vault = st.session_state['vault'][aid]
    meta = vault['meta']
    
    diagnostics = []
    for _, row in results_df.iterrows():
        student_report = {"name": row['Student Name'], "errors": []}
        for q in meta['questions']:
            q_id = f"Q{q['id']}"
            ans = str(row[q_id]).strip().upper()
            if ans != q['correct']:
                student_report['errors'].append({
                    "q": q['id'],
                    "choice": ans,
                    "gap": q['mappings'].get(ans, "Conceptual Error"),
                    "fix": q['remedy']
                })
        diagnostics.append(student_report)
    return diagnostics, None

# --- UI WORKFLOW ---
st.title("🎯 RemediAI Ultra: Multi-Agent Edition")
t1, t2, t3 = st.tabs(["🏗️ Phase 1: Creator", "📤 Phase 2: Upload", "📊 Phase 3: Reports"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        u_school = st.text_input("School Name", "Global International Academy")
        u_topic = st.text_input("Topic", "Photosynthesis & Plant Biology")
        u_grade = st.selectbox("Grade", ["4", "5", "6", "10"])
    with c2:
        u_num = st.number_input("Questions", 1, 15, 5)
        u_time = st.number_input("Time (Mins)", 10, 180, 40)
        u_aid = st.text_input("Assessment ID", value="DIAG-01")

    if st.button("🚀 EXECUTE MULTI-AGENT GENERATION"):
        # Trigger Agent A
        meta = agent_psychometrician(u_topic, u_grade, u_num, 9)
        info = {"aid": u_aid, "grade": u_grade, "topic": u_topic, "time": u_time}
        
        # Save to Vault
        st.session_state['vault'][u_aid] = {"meta": meta, "info": info}
        
        # Trigger Agent B
        pdf = agent_document_architect(meta, info, u_school)
        st.download_button("📥 Download Branded PDF", pdf, f"{u_aid}.pdf")
        
        # Trigger Agent C (Template Generation)
        xl_df = pd.DataFrame(columns=["Student Name"] + [f"Q{i+1}" for i in range(u_num)])
        out = BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            xl_df.to_excel(writer, index=False)
        st.download_button("📥 Download Excel Template", out.getvalue(), f"Template_{u_aid}.xlsx")

with t2:
    st.header("Upload Response Data")
    in_aid = st.text_input("Confirm Assessment ID")
    xl_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if xl_file and in_aid:
        st.success("File Received. Ready for Phase 3.")

with t3:
    st.header("Diagnostic Dashboard")
    active_id = st.selectbox("Select Assessment ID", list(st.session_state['vault'].keys()))
    if active_id and xl_file:
        reports, err = agent_data_engine(active_id, xl_file)
        if err:
            st.error(err)
        else:
            student = st.selectbox("Select Student for Diagnosis", [r['name'] for r in reports])
            rep = next(r for r in reports if r['name'] == student)
            
            if not rep['errors']:
                st.success(f"{student} demonstrated 100% Mastery.")
            for e in rep['errors']:
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #1e3a8a; margin-bottom: 10px;">
                    <b>Question {e['q']} - Incorrect (Selected {e['choice']})</b><br>
                    <i>Detected Misconception:</i> {e['gap']}<br>
                    <b>📍 Remediation:</b> {e['fix']}
                </div>
                """, unsafe_allow_html=True)
