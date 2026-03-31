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
st.set_page_config(page_title="RemediAI Ultra | High-Fidelity Suite", layout="wide")

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

# --- AGENT A: THE PSYCHOMETRICIAN (UPGRADED) ---
def agent_psychometrician(topic, grade, num_q, difficulty):
    # Difficulty context: 1 (Recall), 2 (Application), 3 (Synthesis/Complex Logic)
    prompt = f"""
    Create {num_q} high-fidelity diagnostic MCQs for Grade {grade} on '{topic}'.
    Difficulty Level: {difficulty}/3.
    
    STRICT RULES:
    1. RELEVANCE: Every option (A, B, C, D) must be a plausible answer related to the topic. No 'fillers'.
    2. MISCONCEPTION MAPPING: Every incorrect option must represent a specific logic failure (e.g., 'Reversed Causality', 'Calculation Oversight', 'Context Blindness').
    3. STEM QUALITY: Use scenario-based stems for higher difficulty.
    
    Return JSON: {{'questions': [{{'id':1, 'q':'', 'options':{{'A':'','B':'','C':'','D':''}}, 'correct':'A', 'mappings':{{'B':'Specific Misconception','C':'Specific Misconception','D':'Specific Misconception'}}, 'remedy':''}}]}}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a master of Diagnostic Distractor Engineering."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- AGENT R: THE PEER REVIEWER (NEW) ---
def agent_peer_reviewer(meta, topic):
    # This agent scans the generated questions to ensure they aren't 'hallucinated' or unrelated
    for q in meta['questions']:
        if topic.lower() not in q['q'].lower() and len(q['q']) < 20:
            return False, f"Question {q['id']} seems unrelated or too short."
    return True, "Quality Verified."

# --- AGENT B: DOCUMENT ARCHITECT (PDF) ---
def create_question_paper(metadata, aid, school, topic):
    buf = BytesIO()
    p = canvas.Canvas(buf, pagesize=A4)
    p.setFillColor(colors.HexColor("#1e3a8a"))
    p.rect(0, 750, 600, 100, fill=1, stroke=0)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 18)
    p.drawCentredString(297, 800, school.upper())
    p.setFont("Helvetica", 10)
    p.drawCentredString(297, 780, f"DIAGNOSTIC ASSESSMENT | ID: {aid} | {topic}")
    
    y = 720
    p.setFillColor(colors.black)
    for q in metadata['questions']:
        if y < 150: p.showPage(); y = 800
        p.setFont("Helvetica-Bold", 11)
        p.drawString(50, y, f"Q{q['id']}. {q['q']}")
        y -= 25
        p.setFont("Helvetica", 10)
        for lbl, txt in q['options'].items():
            p.drawString(70, y, f"{lbl}) {txt}")
            y -= 18
        y -= 15
    p.save()
    buf.seek(0)
    return buf

# --- UI WORKFLOW ---
st.title("🎯 RemediAI Ultra: The Institutional Suite")
t1, t2, t3 = st.tabs(["🏗️ Phase 1: Creation", "📤 Phase 2: Ingestion", "📊 Phase 3: Diagnosis"])

with t1:
    c1, c2 = st.columns(2)
    u_school = c1.text_input("School Name", "Vikas International Academy")
    u_topic = c1.text_input("Core Topic", "Animal Cell Structure")
    u_grade = c2.selectbox("Class/Grade", ["6", "7", "8", "9", "10"])
    
    # NEW: DIFFICULTY SCALER
    u_diff = c2.select_slider("Assessment Rigor (Difficulty)", options=[1, 2, 3], value=2)
    
    now = datetime.datetime.now()
    auto_id = f"RAI-{u_grade}-{u_topic[:3].upper()}-{now.strftime('%d%b-%H%M')}"
    u_aid = st.text_input("Auto-Generated ID", value=auto_id)
    u_num = c2.number_input("Question Count", 1, 15, 5)

    if st.button("🚀 INITIATE HIGH-FIDELITY GENERATION"):
        with st.spinner("Psychometrician & Reviewer are engineering the test..."):
            # Step 1: Generate
            meta = agent_psychometrician(u_topic, u_grade, u_num, u_diff)
            
            # Step 2: Peer Review
            is_valid, msg = agent_peer_reviewer(meta, u_topic)
            
            if is_valid:
                # Step 3: Cloud Sync
                table = get_airtable()
                if table:
                    table.create({"Assessment_ID": u_aid, "Metadata": json.dumps(meta), "Topic": u_topic, "School": u_school})
                    st.success(f"Verified & Synced: {u_aid}")
                    
                    # Step 4: Physical Artifacts
                    paper = create_question_paper(meta, u_aid, u_school, u_topic)
                    st.download_button("📥 Download Branded Question Paper", paper, f"Paper_{u_aid}.pdf")
                    
                    xl_df = pd.DataFrame(columns=["Student Name"] + [f"Q{i+1}" for i in range(u_num)])
                    out = BytesIO()
                    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
                        xl_df.to_excel(writer, index=False)
                    st.download_button("📥 Download Data Entry Template", out.getvalue(), f"Template_{u_aid}.xlsx")
            else:
                st.error(f"Quality Check Failed: {msg}. Retrying...")

with t2:
    table = get_airtable()
    if table:
        records = table.all()
        ids = [r['fields']['Assessment_ID'] for r in records if 'Assessment_ID' in r['fields']]
        sel_id = st.selectbox("Select ID from Cloud Archive:", ["Select..."] + list(reversed(ids)))
        if sel_id != "Select...":
            xl_file = st.file_uploader("Upload Response Excel", type=["xlsx"])
            if xl_file:
                rec = next(r for r in records if r['fields'].get('Assessment_ID') == sel_id)
                st.session_state['active_logic'] = json.loads(rec['fields']['Metadata'])
                st.session_state['active_data'] = pd.read_excel(xl_file)
                st.session_state['active_info'] = {"aid": sel_id, "topic": rec['fields'].get('Topic')}
                st.success("Responses Synced to Diagnostic Engine.")

with t3:
    if 'active_logic' in st.session_state:
        st.header("📊 World-Class Diagnostic Dashboard")
        logic = st.session_state['active_logic']
        data = st.session_state['active_data']
        
        # Systemic Analysis
        all_gaps = []
        reports = []
        for _, row in data.iterrows():
            errors = []
            for q in logic['questions']:
                ans = str(row[f"Q{q['id']}"]).strip().upper()
                if ans != q['correct']:
                    gap = q.get('mappings', {}).get(ans, "Conceptual Misunderstanding")
                    errors.append({"q": q['id'], "gap": gap, "fix": q['remedy']})
                    all_gaps.append(gap)
            reports.append({"name": row['Student Name'], "errors": errors})
        
        if all_gaps:
            gap_counts = pd.Series(all_gaps).value_counts()
            st.subheader("🚩 Systemic Learning Gaps")
            st.bar_chart(gap_counts)
        
        st.divider()
        student = st.selectbox("Individual Diagnosis", [r['name'] for r in reports])
        rep = next(r for r in reports if r['name'] == student)
        for e in rep['errors']:
            st.markdown(f"<div class='diagnostic-card'><b>Question {e['q']}</b> - <span style='color:red;'>{e['gap']}</span><br><small><b>Strategic Remedy:</b> {e['fix']}</small></div>", unsafe_allow_html=True)
