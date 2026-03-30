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
st.set_page_config(page_title="RemediAI Ultra | Cloud Suite", layout="wide")

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
    .principal-highlight {
        background-color: #fff1f2; padding: 20px; border-radius: 8px; border-left: 5px solid #e11d48; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Connection Agents
def get_airtable():
    try:
        api = Api(st.secrets["AIRTABLE_TOKEN"])
        return api.table(st.secrets["AIRTABLE_BASE_ID"], st.secrets["AIRTABLE_TABLE_NAME"])
    except Exception as e:
        st.error(f"❌ Airtable Sync Error: {e}")
        return None

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("🔑 OpenAI Key Missing in Secrets!")
    st.stop()

# --- AGENT A: PSYCHOMETRICIAN (DIAGNOSTIC LOGIC) ---
def agent_psychometrician(topic, grade, num_q):
    prompt = f"""
    Create {num_q} deep diagnostic MCQs for Grade {grade} on {topic}. 
    Every wrong option must be a specific 'Smart Mistake' (Misconception).
    Return JSON: {{'questions': [{{'id':1, 'q':'', 'options':{{'A':'','B':'','C':'','D':''}}, 'correct':'A', 'mappings':{{'B':'Reason B','C':'Reason C','D':'Reason D'}}, 'remedy':''}}]}}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master psychometrician like EI ASSET."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT B: DOCUMENT ARCHITECT (PDF) ---
def create_pdf(metadata, aid, school, topic):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    # Header Branding
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 20)
    p.drawCentredString(297, 800, school.upper())
    p.setFont("Helvetica", 10)
    p.drawCentredString(297, 780, f"DIAGNOSTIC ASSESSMENT | ID: {aid} | Topic: {topic}")
    
    y = 720
    for q in metadata['questions']:
        if y < 120: p.showPage(); y = 800
        p.setFillColor(colors.black)
        p.setFont("Helvetica-Bold", 11)
        # Simple wrapping
        q_text = q.get('q') or q.get('question')
        p.drawString(50, y, f"Q{q['id']}. {q_text[:85]}")
        y -= 25
        p.setFont("Helvetica", 10)
        for l, t in q['options'].items():
            p.drawString(75, y, f"{l}) {t}")
            y -= 18
        y -= 15
    p.save()
    buf.seek(0)
    return buf

# --- UI WORKFLOW ---
st.title("🚀 RemediAI Ultra: Universal Cloud Suite")
t1, t2, t3 = st.tabs(["🏗️ Phase 1: Create", "📤 Phase 2: Upload", "📊 Phase 3: Reports"])

with t1:
    c1, c2 = st.columns(2)
    u_school = c1.text_input("School Name", "Global International Academy")
    u_topic = c1.text_input("Topic", "A Letter to God")
    u_grade = c2.selectbox("Grade", ["9", "10", "11"])
    u_aid = c2.text_input("Assessment ID", value="DIAG-101")
    u_num = c2.number_input("Question Count", 1, 15, 5)

    if st.button("🚀 GENERATE & SYNC TO CLOUD"):
        with st.spinner("Psychometric AI is mapping the diagnostic logic..."):
            meta = agent_psychometrician(u_topic, u_grade, u_num)
            table = get_airtable()
            if table:
                table.create({
                    "Assessment_ID": u_aid,
                    "Metadata": json.dumps(meta),
                    "Topic": u_topic,
                    "School": u_school
                })
                st.success(f"Assessment {u_aid} successfully saved to Cloud.")
                st.download_button("📥 Download Branded Paper", create_pdf(meta, u_aid, u_school, u_topic), f"{u_aid}.pdf")
                
                # Excel Template
                xl_df = pd.DataFrame(columns=["Student Name"] + [f"Q{i+1}" for i in range(u_num)])
                out = BytesIO()
                with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                    xl_df.to_excel(writer, index=False)
                st.download_button("📥 Download Excel Template", out.getvalue(), f"Template_{u_aid}.xlsx")

with t2:
    st.header("Phase 2: Universal Data Ingestion")
    table = get_airtable()
    if table:
        records = table.all()
        ids = [r['fields']['Assessment_ID'] for r in records if 'Assessment_ID' in r['fields']]
        sel_id = st.selectbox("Select Assessment ID from Cloud Archive:", ["Select..."] + ids)
        if sel_id != "Select...":
            xl_file = st.file_uploader("Upload Responses", type=["xlsx"])
            if xl_file:
                rec = next(r for r in records if r['fields'].get('Assessment_ID') == sel_id)
                st.session_state['active_logic'] = json.loads(rec['fields']['Metadata'])
                st.session_state['active_data'] = pd.read_excel(xl_file)
                st.success(f"Linked results to logic for {sel_id}")

with t3:
    if 'active_logic' in st.session_state:
        st.header("📊 Institutional Diagnostic Dashboard")
        logic = st.session_state['active_logic']
        data = st.session_state['active_data']
        
        # --- CLASS GAP ANALYSIS ---
        st.subheader("🚩 Class-Wide Thinking Gaps")
        all_gaps = []
        for _, row in data.iterrows():
            for q in logic['questions']:
                ans = str(row[f"Q{q['id']}"]).strip().upper()
                if ans != q['correct']:
                    gap = q.get('mappings', {}).get(ans, "Thinking Error")
                    all_gaps.append(gap)
        
        if all_gaps:
            gap_counts = pd.Series(all_gaps).value_counts()
            st.bar_chart(gap_counts)
            st.markdown(f"""
            <div class="principal-highlight">
                <h4 style="margin:0; color: #9f1239;">SYSTEMIC ISSUE: {gap_counts.index[0]}</h4>
                <p>Action: Reteach the logic behind this specific misconception.</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        student = st.selectbox("Detailed Student Diagnosis", data['Student Name'].unique())
        s_row = data[data['Student Name'] == student].iloc[0]
        
        for q in logic['questions']:
            ans = str(s_row[f"Q{q['id']}"]).strip().upper()
            if ans != q['correct']:
                st.markdown(f"<div class='diagnostic-card'><b>Q{q['id']} - Incorrect</b><br><i>Gap:</i> {q.get('mappings', {}).get(ans, 'Thinking Error')}<br><small><b>Remedy:</b> {q['remedy']}</small></div>", unsafe_allow_html=True)
            else:
                st.success(f"Question {q['id']} - Mastered")
