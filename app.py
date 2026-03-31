import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import datetime
from io import BytesIO
from pyairtable import Api
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# --- 1. CORE SYSTEM SETUP ---
st.set_page_config(page_title="RemediAI Ultra | Executive Suite", layout="wide")

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
    try:
        api = Api(st.secrets["AIRTABLE_TOKEN"])
        return api.table(st.secrets["AIRTABLE_BASE_ID"], st.secrets["AIRTABLE_TABLE_NAME"])
    except: return None

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- AGENT A: PSYCHOMETRICIAN ---
def agent_psychometrician(topic, grade, num_q):
    prompt = f"""
    Create {num_q} deep diagnostic MCQs for Grade {grade} on {topic}. 
    Every wrong option must be a specific 'Thinking Error' or Misconception.
    Return JSON: {{'questions': [{{'id':1, 'q':'', 'options':{{'A':'','B':'','C':'','D':''}}, 'correct':'A', 'mappings':{{'B':'Misconception X','C':'Misconception Y','D':'Misconception Z'}}, 'remedy':''}}]}}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master psychometrician for diagnostic assessments."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT B: DOCUMENT ARCHITECT (QUESTION PAPER) ---
def create_question_paper(metadata, aid, school, topic):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(297, 800, school.upper())
    p.setFont("Helvetica", 10)
    p.drawCentredString(297, 780, f"DIAGNOSTIC TEST | ID: {aid} | TOPIC: {topic}")
    
    y = 720
    p.setFillColor(colors.black)
    for q in metadata['questions']:
        if y < 150: p.showPage(); y = 800
        p.setFont("Helvetica-Bold", 11)
        p.drawString(50, y, f"Q{q['id']}. {q.get('q') or q.get('question')}")
        y -= 25
        p.setFont("Helvetica", 10)
        opts = q.get('options', {})
        for lbl in ["A", "B", "C", "D"]:
            p.drawString(70, y, f"{lbl}) {opts.get(lbl, '---')}")
            y -= 18
        y -= 15
    p.save()
    buf.seek(0)
    return buf

# --- AGENT H: THE VISUAL ANALYST (PDF CLASS REPORT) ---
def create_class_report(info, gap_counts):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setFillColor(colors.HexColor("#9f1239")) # Crimson for Leadership
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, "EXECUTIVE CLASS MISCONCEPTION SUMMARY")
    p.setFont("Helvetica", 10)
    p.drawString(50, 780, f"ID: {info['aid']} | Topic: {info['topic']}")
    
    y = 700
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, y, "Top Systemic Thinking Gaps Found:")
    y -= 40
    
    for gap, count in gap_counts.items():
        if y < 100: p.showPage(); y = 800
        # Draw Frequency Bar in PDF
        bar_width = min(count * 20, 300)
        p.setFillColor(colors.lightgrey)
        p.rect(150, y-5, 300, 15, fill=1, stroke=0)
        p.setFillColor(colors.HexColor("#9f1239"))
        p.rect(150, y-5, bar_width, 15, fill=1, stroke=0)
        
        p.setFillColor(colors.black)
        p.setFont("Helvetica-Bold", 10)
        p.drawString(50, y, gap[:18])
        p.drawRightString(540, y, f"{count} Students")
        y -= 30
        
    p.save(); buf.seek(0)
    return buf

# --- AGENT I: STUDENT DIAGNOSTICIAN (PDF INDIVIDUAL) ---
def create_individual_report(student, errors, info):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, f"STUDENT MISCONCEPTION PROFILE: {student.upper()}")
    p.setFont("Helvetica", 10)
    p.drawString(50, 780, f"ID: {info['aid']} | Topic: {info['topic']}")
    
    y = 720
    p.setFillColor(colors.black)
    if not errors:
        p.setFont("Helvetica-Bold", 14); p.drawString(50, y, "NO CONCEPTUAL GAPS DETECTED.")
    else:
        for e in errors:
            if y < 120: p.showPage(); y = 800
            p.setFont("Helvetica-Bold", 11); p.setFillColor(colors.HexColor("#1e3a8a"))
            p.drawString(50, y, f"Question {e['q']} - Misconception Analysis")
            y -= 15
            p.setFont("Helvetica-Bold", 10); p.setFillColor(colors.red)
            p.drawString(60, y, f"Thinking Error: {e['gap']}")
            y -= 15
            p.setFillColor(colors.black); p.setFont("Helvetica", 10)
            p.drawString(60, y, f"Discussion/Remedy: {e['fix']}")
            y -= 35
    p.save(); buf.seek(0)
    return buf

# --- UI WORKFLOW ---
st.title("🎯 RemediAI Ultra: Universal Institutional Suite")
t1, t2, t3 = st.tabs(["🏗️ Phase 1: Create", "📤 Phase 2: Upload", "📊 Phase 3: Reports"])

with t1:
    c1, c2 = st.columns(2)
    u_school = c1.text_input("School Name", "Global International Academy")
    u_topic = c1.text_input("Topic", "A Letter to God")
    u_grade = c2.selectbox("Grade", ["9", "10", "11"])
    
    # Automated ID
    now = datetime.datetime.now()
    auto_id = f"RAI-{u_grade}-{u_topic[:3].upper()}-{now.strftime('%d%b-%H%M')}"
    u_aid = st.text_input("Unique Assessment ID", value=auto_id)
    u_num = c2.number_input("Question Count", 1, 15, 5)

    if st.button("🚀 GENERATE & SYNC TO CLOUD"):
        with st.spinner("Synthesizing diagnostic logic..."):
            meta = agent_psychometrician(u_topic, u_grade, u_num)
            table = get_airtable()
            if table:
                table.create({"Assessment_ID": u_aid, "Metadata": json.dumps(meta), "Topic": u_topic, "School": u_school})
                st.success(f"Logic for {u_aid} is now archived in the cloud.")
                
                # Immediate Downloads
                paper = create_question_paper(meta, u_aid, u_school, u_topic)
                st.download_button("📥 Download Student Question Paper", paper, f"Paper_{u_aid}.pdf")
                
                xl_df = pd.DataFrame(columns=["Student Name"] + [f"Q{i+1}" for i in range(u_num)])
                out = BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                    xl_df.to_excel(writer, index=False)
                st.download_button("📥 Download Excel Data Template", out.getvalue(), f"Template_{u_aid}.xlsx")

with t2:
    table = get_airtable()
    if table:
        records = table.all()
        ids = [r['fields']['Assessment_ID'] for r in records if 'Assessment_ID' in r['fields']]
        sel_id = st.selectbox("Select Cloud ID", ["Select..."] + list(reversed(ids)))
        if sel_id != "Select...":
            xl_file = st.file_uploader("Upload Responses", type=["xlsx"])
            if xl_file:
                rec = next(r for r in records if r['fields'].get('Assessment_ID') == sel_id)
                st.session_state['active_logic'] = json.loads(rec['fields']['Metadata'])
                st.session_state['active_data'] = pd.read_excel(xl_file)
                st.session_state['active_info'] = {"aid": sel_id, "topic": rec['fields'].get('Topic')}
                st.success("Cloud Data Linked.")

with t3:
    if 'active_logic' in st.session_state:
        st.header("📊 Executive Diagnostic Hub")
        logic = st.session_state['active_logic']
        data = st.session_state['active_data']
        info = st.session_state['active_info']
        
        all_gaps = []
        reports = []
        for _, row in data.iterrows():
            errors = []
            for q in logic['questions']:
                ans = str(row[f"Q{q['id']}"]).strip().upper()
                if ans != q['correct']:
                    gap = q.get('mappings', {}).get(ans, "Conceptual Error")
                    errors.append({"q": q['id'], "gap": gap, "fix": q['remedy']})
                    all_gaps.append(gap)
            reports.append({"name": row['Student Name'], "errors": errors})
        
        if all_gaps:
            gap_counts = pd.Series(all_gaps).value_counts()
            st.bar_chart(gap_counts)
            st.download_button("📥 Download Class Misconception Summary", create_class_report(info, gap_counts), "Class_Report.pdf")
        
        st.divider()
        student = st.selectbox("Select Student Profile", [r['name'] for r in reports])
        rep = next(r for r in reports if r['name'] == student)
        st.download_button(f"📥 Download Report for {student}", create_individual_report(student, rep['errors'], info), f"{student}.pdf")
        
        for e in rep['errors']:
            st.markdown(f"<div class='diagnostic-card'><b>Q{e['q']}</b> - <span style='color:red;'>{e['gap']}</span><br><small>{e['fix']}</small></div>", unsafe_allow_html=True)
