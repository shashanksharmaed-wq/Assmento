import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# --- 1. CORE SYSTEM SETUP ---
st.set_page_config(page_title="RemediAI Ultra | Multi-Agent Diagnostic", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { 
        border-radius: 8px; background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); 
        color: white; font-weight: bold; height: 3.5em; width: 100%; border: none;
    }
    .diagnostic-card { 
        background-color: white; padding: 20px; border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-left: 6px solid #1e3a8a; margin-bottom: 15px;
    }
    .executive-banner {
        background-color: #fff1f2; border-left: 5px solid #e11d48; padding: 20px; border-radius: 8px; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("🔑 OpenAI API Key Missing!")
    st.stop()

if 'vault' not in st.session_state: st.session_state['vault'] = {}

# --- AGENT A: THE PSYCHOMETRICIAN ---
def agent_psychometrician(topic, grade, num_q, diff):
    prompt = f"Create {num_q} 'Deep Diagnostic' MCQs for Grade {grade} on {topic}. Difficulty {diff}/12. Options must be misconception-driven. Return JSON with questions, options, correct, mappings, and remedy."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master of educational diagnostic design like EI ASSET."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT B: THE DOCUMENT ARCHITECT (PAPER) ---
def agent_document_architect(metadata, info, school_name):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, h-90, w, 90, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(w/2, h-40, school_name.upper())
    p.setFont("Helvetica", 10)
    p.drawCentredString(w/2, h-60, f"DIAGNOSTIC ASSESSMENT | ID: {info['aid']}")
    
    y = h-130
    for q in metadata.get('questions', []):
        if y < 150: p.showPage(); y = h-50
        p.setFont("Helvetica-Bold", 11)
        p.setFillColor(colors.black)
        p.drawString(50, y, f"Q{q['id']}. {q.get('q') or q.get('question')}")
        y -= 25
        opts = q.get('options', {})
        for lbl in ["A", "B", "C", "D"]:
            p.drawString(75, y, f"{lbl}) {opts.get(lbl)}")
            y -= 15
        y -= 20
    p.save()
    buffer.seek(0)
    return buffer

# --- AGENT F: THE EXECUTIVE DEAN (PRINCIPAL'S SUMMARY) ---
def agent_executive_dean(reports, info, school_name):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    
    # Elegant Branding
    p.setFillColor(colors.HexColor("#9f1239")) # Crimson Theme for Leadership
    p.rect(0, h-100, w, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(w/2, h-45, f"{school_name.upper()} - EXECUTIVE SUMMARY")
    p.setFont("Helvetica", 12)
    p.drawCentredString(w/2, h-70, f"Diagnostic Assessment: {info['topic']}")
    
    # Accuracy Stats
    total_q = len(reports[0]['errors']) + 1 # Rough estimate
    correct_total = sum([1 for r in reports if not r['errors']])
    
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, h-140, "Institutional Health Overview")
    p.setFont("Helvetica", 11)
    p.drawString(50, h-160, f"Assessment ID: {info['aid']}  |  Grade: {info['grade']}")
    p.drawString(50, h-175, f"Overall Class Proficiency: {int((correct_total/len(reports))*100)}%")
    
    # Systemic Gaps
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, h-210, "Top Systemic Misconceptions (Action Items)")
    y = h-235
    
    clusters = {}
    for r in reports:
        for e in r['errors']:
            gap = e['gap']
            clusters[gap] = clusters.get(gap, 0) + 1
            
    for gap, count in sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:3]:
        p.setFont("Helvetica-Bold", 11)
        p.drawString(60, y, f"• {gap}")
        p.setFont("Helvetica", 10)
        p.drawString(70, y-15, f"Frequency: {count} students affected. Strategy: Remedial intervention required.")
        y -= 40

    p.save()
    buffer.seek(0)
    return buffer

# --- APP FLOW ---
st.title("🎯 RemediAI Ultra: The Multi-Agent Diagnostic Engine")
t1, t2, t3 = st.tabs(["🏗️ Creator", "📤 Upload", "📊 Strategic Dashboard"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        u_school = st.text_input("School Name", "Global International Academy")
        u_topic = st.text_input("Topic", "Photosynthesis")
        u_grade = st.selectbox("Grade", ["9", "10", "11"])
    with c2:
        u_num = st.number_input("Questions", 1, 15, 5)
        u_aid = st.text_input("Assessment ID", value="DIAG-01")

    if st.button("🚀 EXECUTE FULL GENERATION"):
        meta = agent_psychometrician(u_topic, u_grade, u_num, 9)
        info = {"aid": u_aid, "grade": u_grade, "topic": u_topic}
        st.session_state['vault'][u_aid] = {"meta": meta, "info": info}
        
        pdf = agent_document_architect(meta, info, u_school)
        st.download_button("📥 Download Branded Question Paper", pdf, f"{u_aid}.pdf")
        
        xl_df = pd.DataFrame(columns=["Student Name"] + [f"Q{i+1}" for i in range(u_num)])
        out = BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            xl_df.to_excel(writer, index=False)
        st.download_button("📥 Download Excel Response Template", out.getvalue(), f"Template_{u_aid}.xlsx")

with t2:
    in_aid = st.text_input("Enter Assessment ID")
    xl_file = st.file_uploader("Upload Excel", type=["xlsx"])
    if xl_file and in_aid:
        st.success("Responses ready for strategic analysis.")

with t3:
    if 'xl_file' in locals() and xl_file and in_aid in st.session_state['vault']:
        df = pd.read_excel(xl_file)
        meta = st.session_state['vault'][in_aid]['meta']
        info = st.session_state['vault'][in_aid]['info']
        
        reports = []
        for _, row in df.iterrows():
            errors = []
            for q in meta['questions']:
                ans = str(row[f"Q{q['id']}"]).strip().upper()
                if ans != q['correct']:
                    errors.append({"q": q['id'], "gap": q['mappings'].get(ans, "Logic Gap"), "fix": q['remedy']})
            reports.append({"name": row['Student Name'], "errors": errors})
        
        # Principal's Summary Button
        summary_pdf = agent_executive_dean(reports, info, u_school)
        st.download_button("👑 Download Principal's Executive Summary (PDF)", summary_pdf, f"Principal_Report_{in_aid}.pdf")
        
        # UI Visuals
        st.divider()
        student = st.selectbox("Detailed Student Diagnosis", [r['name'] for r in reports])
        rep = next(r for r in reports if r['name'] == student)
        for e in rep['errors']:
            st.markdown(f"<div class='diagnostic-card'><b>Q{e['q']}:</b> {e['gap']}<br><small>{e['fix']}</small></div>", unsafe_allow_html=True)
