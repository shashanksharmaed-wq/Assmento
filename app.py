import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os
from io import BytesIO
from pyairtable import Api
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# --- 1. SYSTEM SETUP & CLOUD SYNC ---
st.set_page_config(page_title="RemediAI Ultra | Universal Sync", layout="wide")

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
    </style>
    """, unsafe_allow_html=True)

# Connection Agents
def get_airtable():
    api = Api(st.secrets["AIRTABLE_TOKEN"])
    return api.table(st.secrets["AIRTABLE_BASE_ID"], st.secrets["AIRTABLE_TABLE_NAME"])

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("🔑 OpenAI API Key Missing!")
    st.stop()

# --- AGENT A: THE PSYCHOMETRICIAN (DIAGNOSTIC LOGIC) ---
def agent_psychometrician(topic, grade, num_q):
    prompt = f"Create {num_q} deep diagnostic MCQs for Grade {grade} on {topic}. Every wrong option must be a specific misconception. Return JSON with questions, options, correct, mappings, and remedy."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master psychometrician for EI ASSET."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT B: THE DOCUMENT ARCHITECT (BRANDED PDF) ---
def agent_document_architect(metadata, info, school_name):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    
    # Header Banner (Deep Blue)
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
        q_text = q.get('q') or q.get('question') or "N/A"
        p.drawString(50, y, f"Q{q['id']}. {q_text}")
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
    p.setFillColor(colors.HexColor("#9f1239"))
    p.rect(0, h-100, w, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(w/2, h-45, f"{school_name.upper()} - EXECUTIVE SUMMARY")
    p.save()
    buffer.seek(0)
    return buffer

# --- UI WORKFLOW ---
st.title("🚀 RemediAI Ultra: Universal Institutional Suite")
t1, t2, t3 = st.tabs(["🏗️ Phase 1: Create", "📤 Phase 2: Upload", "📊 Phase 3: Reports"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        u_school = st.text_input("School Name", "Global International Academy")
        u_topic = st.text_input("Topic", "A Letter to God")
        u_grade = st.selectbox("Grade", ["9", "10", "11"])
    with c2:
        u_num = st.number_input("No. of Questions", 1, 15, 5)
        u_aid = st.text_input("Assessment ID", value="DIAG-101")

    if st.button("🚀 GENERATE & SYNC TO CLOUD"):
        with st.spinner("Engineering high-fidelity misconceptions..."):
            meta = agent_psychometrician(u_topic, u_grade, u_num)
            info = {"aid": u_aid, "grade": u_grade, "topic": u_topic}
            
            # SAVE TO AIRTABLE (Universal Sync)
            table = get_airtable()
            table.create({
                "Assessment_ID": u_aid,
                "Metadata": json.dumps(meta),
                "Topic": u_topic,
                "School": u_school
            })
            
            st.success(f"Assessment {u_aid} is now archived in the Cloud.")
            
            pdf = agent_document_architect(meta, info, u_school)
            st.download_button("📥 Download Branded Paper", pdf, f"{u_aid}.pdf")
            
            xl_df = pd.DataFrame(columns=["Student Name"] + [f"Q{i+1}" for i in range(u_num)])
            out = BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                xl_df.to_excel(writer, index=False)
            st.download_button("📥 Download Response Template", out.getvalue(), f"Template_{u_aid}.xlsx")

with t2:
    st.header("Upload Results for Any Cloud-Stored ID")
    table = get_airtable()
    records = table.all()
    ids = [r['fields']['Assessment_ID'] for r in records]
    
    sel_id = st.selectbox("Select Assessment ID from Cloud Archive:", ["Select..."] + ids)
    if sel_id != "Select...":
        xl_file = st.file_uploader("Upload Excel Responses", type=["xlsx"])
        if xl_file:
            rec = next(r for r in records if r['fields']['Assessment_ID'] == sel_id)
            st.session_state['active_logic'] = json.loads(rec['fields']['Metadata'])
            st.session_state['active_data'] = pd.read_excel(xl_file)
            st.success(f"Linked results to logic for {sel_id}")

with t3:
    if 'active_logic' in st.session_state:
        st.header("Strategic Diagnostic Hub")
        logic = st.session_state['active_logic']
        data = st.session_state['active_data']
        
        student = st.selectbox("Select Student", data['Student Name'].unique())
        s_row = data[data['Student Name'] == student].iloc[0]
        
        for q in logic['questions']:
            ans = str(s_row[f"Q{q['id']}"]).strip().upper()
            if ans != q['correct']:
                mapping = q.get('mappings') or q.get('engine') or {}
                st.markdown(f"<div class='diagnostic-card'><b>Q{q['id']}:</b> {mapping.get(ans, 'Logic Error')}<br><small>{q['remedy']}</small></div>", unsafe_allow_html=True)
            else:
                st.success(f"Q{q['id']} Mastered")
