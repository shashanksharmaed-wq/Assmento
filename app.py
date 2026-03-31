import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from io import BytesIO
from pyairtable import Api
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# --- 1. CORE SYSTEM SETUP ---
st.set_page_config(page_title="RemediAI Ultra | Professional Portfolio", layout="wide")

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

# Connection Agents
def get_airtable():
    try:
        api = Api(st.secrets["AIRTABLE_TOKEN"])
        return api.table(st.secrets["AIRTABLE_BASE_ID"], st.secrets["AIRTABLE_TABLE_NAME"])
    except: return None

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- AGENT A: PSYCHOMETRICIAN ---
def agent_psychometrician(topic, grade, num_q):
    prompt = f"Create {num_q} deep diagnostic MCQs for Grade {grade} on {topic}. Wrong options must be specific misconceptions. Return JSON with questions, options, correct, mappings, and remedy."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master psychometrician for EI ASSET."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT G: PORTFOLIO GENERATOR (PDF REPORTS) ---

def create_class_report(info, gap_counts):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, f"CLASS-WIDE MISCONCEPTION REPORT")
    p.setFont("Helvetica", 10)
    p.drawString(50, 780, f"Topic: {info['topic']} | Assessment ID: {info['aid']}")
    
    y = 700
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Top Systemic Gaps Found in Class:")
    y -= 30
    
    for gap, count in gap_counts.items():
        if y < 100: p.showPage(); y = 800
        p.setFont("Helvetica-Bold", 11)
        p.drawString(60, y, f"• {gap}")
        y -= 15
        p.setFont("Helvetica", 10)
        p.drawString(70, y, f"Frequency: {count} students affected.")
        y -= 25
    
    p.save()
    buf.seek(0)
    return buf

def create_individual_report(student_name, errors, info):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, f"STUDENT DIAGNOSTIC: {student_name.upper()}")
    p.setFont("Helvetica", 10)
    p.drawString(50, 780, f"Topic: {info['topic']} | ID: {info['aid']}")
    
    y = 720
    p.setFillColor(colors.black)
    if not errors:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y, "RESULT: 100% CONCEPTUAL MASTERY")
    else:
        for e in errors:
            if y < 120: p.showPage(); y = 800
            p.setFont("Helvetica-Bold", 11)
            p.drawString(50, y, f"Question {e['q']} - Misconception Analysis:")
            y -= 15
            p.setFont("Helvetica-Bold", 10)
            p.drawString(60, y, f"Thinking Error: {e['gap']}")
            y -= 15
            p.setFont("Helvetica", 10)
            p.drawString(60, y, f"Remedial Advice: {e['fix']}")
            y -= 35
    p.save()
    buf.seek(0)
    return buf

# --- UI WORKFLOW ---
st.title("🎯 RemediAI Ultra: The Institutional Suite")
t1, t2, t3 = st.tabs(["🏗️ Phase 1: Create", "📤 Phase 2: Upload", "📊 Phase 3: Reports"])

with t1:
    c1, c2 = st.columns(2)
    u_school = c1.text_input("School Name", "Global International Academy")
    u_topic = c1.text_input("Topic", "A Letter to God")
    u_grade = c2.selectbox("Grade", ["9", "10", "11"])
    u_aid = c2.text_input("Assessment ID", value="DIAG-101")
    u_num = c2.number_input("Question Count", 1, 15, 5)

    if st.button("🚀 GENERATE & SYNC"):
        meta = agent_psychometrician(u_topic, u_grade, u_num)
        table = get_airtable()
        if table:
            table.create({"Assessment_ID": u_aid, "Metadata": json.dumps(meta), "Topic": u_topic, "School": u_school})
            st.success(f"Logic for {u_aid} synced.")

with t2:
    table = get_airtable()
    if table:
        records = table.all()
        ids = [r['fields']['Assessment_ID'] for r in records if 'Assessment_ID' in r['fields']]
        sel_id = st.selectbox("Select Cloud ID", ["Select..."] + ids)
        if sel_id != "Select...":
            xl_file = st.file_uploader("Upload Excel Responses", type=["xlsx"])
            if xl_file:
                rec = next(r for r in records if r['fields'].get('Assessment_ID') == sel_id)
                st.session_state['active_logic'] = json.loads(rec['fields']['Metadata'])
                st.session_state['active_data'] = pd.read_excel(xl_file)
                st.session_state['active_info'] = {"aid": sel_id, "topic": rec['fields'].get('Topic')}

with t3:
    if 'active_logic' in st.session_state:
        logic = st.session_state['active_logic']
        data = st.session_state['active_data']
        info = st.session_state['active_info']
        
        # Process Reports
        reports = []
        all_gaps = []
        for _, row in data.iterrows():
            errors = []
            for q in logic['questions']:
                ans = str(row[f"Q{q['id']}"]).strip().upper()
                if ans != q['correct']:
                    gap = q.get('mappings', {}).get(ans, "Logic Error")
                    errors.append({"q": q['id'], "gap": gap, "fix": q['remedy']})
                    all_gaps.append(gap)
            reports.append({"name": row['Student Name'], "errors": errors})
        
        st.header("📊 Class-Wide Misconception Report")
        if all_gaps:
            gap_counts = pd.Series(all_gaps).value_counts()
            st.bar_chart(gap_counts)
            
            # --- DOWNLOAD CLASS REPORT ---
            class_pdf = create_class_report(info, gap_counts)
            st.download_button("📥 Download Class Misconception Report (PDF)", class_pdf, f"Class_Diagnosis_{info['aid']}.pdf")
            
            st.subheader("🤝 Remedial Grouping")
            clusters = {}
            for r in reports:
                for e in r['errors']:
                    clusters[e['gap']] = clusters.get(e['gap'], []) + [r['name']]
            
            for gap, students in clusters.items():
                with st.expander(f"🔴 Group: {gap} ({len(students)} Students)"):
                    st.write(f"**Discussion Focus:** {gap}")
                    st.write(f"**Students:** {', '.join(students)}")

        st.divider()

        # --- INDIVIDUAL REPORT SECTION ---
        st.subheader("👤 Individual Student Portfolios")
        student = st.selectbox("Select Student", [r['name'] for r in reports])
        rep = next(r for r in reports if r['name'] == student)
        
        ind_pdf = create_individual_report(student, rep['errors'], info)
        st.download_button(f"📥 Download Student Misconception Report for {student}", ind_pdf, f"{student}_Diagnosis.pdf")
        
        for e in rep['errors']:
            st.markdown(f"<div class='diagnostic-card'><b>Q{e['q']}</b><br><i>Misconception:</i> {e['gap']}<br><small>{e['fix']}</small></div>", unsafe_allow_html=True)
