import streamlit as st
import openai
import pandas as pd
import sqlite3
import json
import uuid
import re
import time
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Academic Diagnostic Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
LLM_MODEL = "gpt-4.1"

# =====================================================
# LOAD SYLLABUS (MINIMUM ASSUMPTION)
# =====================================================

@st.cache_data
def load_syllabus():
    df = pd.read_csv("Teachshank_Master_Database_FINAL.tsv", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_syllabus()

CLASS_COL = next((c for c in df.columns if "class" in c.lower() or "grade" in c.lower()), None)
SUBJECT_COL = next((c for c in df.columns if "subject" in c.lower()), None)
TOPIC_COL = next((c for c in df.columns if c not in [CLASS_COL, SUBJECT_COL]), None)

if not CLASS_COL or not SUBJECT_COL:
    st.error("Syllabus file must contain Grade/Class and Subject columns.")
    st.stop()

# =====================================================
# DATABASE
# =====================================================

conn = sqlite3.connect("assessment.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT,
    grade TEXT,
    subject TEXT,
    topic TEXT,
    score INTEGER,
    total INTEGER,
    created_at TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS responses (
    attempt_id TEXT,
    qno INTEGER,
    question TEXT,
    selected TEXT,
    correct TEXT,
    explanation TEXT
)
""")
conn.commit()

# =====================================================
# SAFE JSON EXTRACTION
# =====================================================

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None

# =====================================================
# HARD QUESTION GENERATOR
# =====================================================

def generate_question(context, retries=3):
    prompt = f"""
Create ONE HARD diagnostic multiple-choice question.

Context:
{context}

MANDATORY RULES:
- Minimum TWO reasoning steps
- No direct formula substitution
- At least TWO misconception-based distractors
- All options must look equally plausible

Return JSON ONLY:
{{
 "question": "",
 "options": ["A","B","C","D"],
 "answer": "",
 "explanation": "Explain why each wrong option is tempting but incorrect"
}}
"""
    for _ in range(retries):
        r = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        data = extract_json(r.choices[0].message.content)
        if data and "question" in data:
            return data

    return {
        "question": "Question generation failed. Please retry.",
        "options": ["A","B","C","D"],
        "answer": "A",
        "explanation": "System fallback."
    }

# =====================================================
# UI – CLEAN ASSESSMENT INDEX
# =====================================================

st.title("🧠 Academic Diagnostic Assessment")
st.caption("Reasoning • Conceptual Understanding • Misconception Analysis")

with st.sidebar:
    st.markdown("### 📘 Assessment Setup")

    grades = sorted(df[CLASS_COL].dropna().unique())
    grade = st.selectbox("Grade", grades)

    subjects = sorted(df[df[CLASS_COL] == grade][SUBJECT_COL].dropna().unique())
    subject = st.selectbox("Subject", subjects)

    if TOPIC_COL:
        topics = sorted(
            df[(df[CLASS_COL] == grade) & (df[SUBJECT_COL] == subject)][TOPIC_COL]
            .dropna().unique()
        )
        topic = st.selectbox("Topic", topics)
    else:
        topic = "General Concepts"

    num_q = st.slider("Number of Questions", 5, 15, 8)
    duration = st.slider("Time (minutes)", 10, 45, 20)
    start = st.button("▶ Start Assessment")

# =====================================================
# START ASSESSMENT
# =====================================================

if start:
    st.session_state.attempt_id = str(uuid.uuid4())
    st.session_state.questions = []
    st.session_state.responses = {}
    st.session_state.start_time = time.time()
    st.session_state.duration = duration * 60

    context = f"""
Grade: {grade}
Subject: {subject}
Topic: {topic}
"""

    for _ in range(num_q):
        st.session_state.questions.append(generate_question(context))

# =====================================================
# ASSESSMENT VIEW (WITH TIMER)
# =====================================================

if "questions" in st.session_state:
    remaining = int(st.session_state.duration - (time.time() - st.session_state.start_time))

    if remaining <= 0:
        st.warning("⏰ Time is over. Submitting automatically.")
        submit = True
    else:
        mins, secs = divmod(remaining, 60)
        st.info(f"⏱ Time Remaining: {mins:02d}:{secs:02d}")
        submit = False

    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        st.session_state.responses[i] = st.radio(
            "",
            q["options"],
            key=f"q_{i}",
            index=None
        )
        st.markdown("---")

    if st.button("Submit Assessment"):
        submit = True

# =====================================================
# RESULTS + REPORT + PERCENTILE
# =====================================================

if "questions" in st.session_state and submit:
    score = 0

    for i, q in enumerate(st.session_state.questions):
        sel = st.session_state.responses.get(i)
        if sel == q["answer"]:
            score += 1

        cur.execute("""
        INSERT INTO responses VALUES (?,?,?,?,?,?)
        """, (
            st.session_state.attempt_id,
            i + 1,
            q["question"],
            sel,
            q["answer"],
            q["explanation"]
        ))

    cur.execute("""
    INSERT INTO attempts VALUES (?,?,?,?,?,?)
    """, (
        st.session_state.attempt_id,
        str(grade),
        subject,
        topic,
        score,
        len(st.session_state.questions),
        datetime.now().isoformat()
    ))
    conn.commit()

    st.markdown("## 📊 Diagnostic Report")
    st.markdown(f"**Score:** {score} / {len(st.session_state.questions)}")

    cur.execute("""
    SELECT score FROM attempts
    WHERE grade=? AND subject=? AND topic=?
    """, (str(grade), subject, topic))
    scores = [r[0] for r in cur.fetchall()]
    percentile = round(100 * sum(s < score for s in scores) / max(len(scores), 1), 1)

    st.markdown(f"**Percentile (within this group):** {percentile}%")

    st.markdown("""
**Interpretation**
- High score → Strong conceptual clarity  
- Medium score → Partial misconceptions  
- Low score → Foundational gaps identified  
""")

    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"### Q{i+1}")
        st.write(q["question"])
        st.write(f"**Your Answer:** {st.session_state.responses.get(i)}")
        st.write(f"**Correct Answer:** {q['answer']}")
        st.write(f"**Explanation:** {q['explanation']}")
        st.markdown("---")

    st.markdown("""
### 🏫 Why Schools Use This Assessment
• Fresh questions generated every time  
• No repetition across attempts  
• Focus on reasoning and misconceptions  
• Benchmarking using percentiles  
• Suitable for grades 9–12  
""")
