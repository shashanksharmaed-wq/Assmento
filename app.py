import streamlit as st
import openai
import pandas as pd
import sqlite3
import json
import uuid
import re
from datetime import datetime

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Diagnostic Assessment (EI-Style)",
    layout="wide",
    initial_sidebar_state="expanded"
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
LLM_MODEL = "gpt-4.1"

# =========================
# LOAD TSV (MINIMUM ASSUMPTION)
# =========================

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
    st.error("Syllabus file must contain Class/Grade and Subject columns.")
    st.stop()

# =========================
# DATABASE
# =========================

conn = sqlite3.connect("attempts.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT,
    qno INTEGER,
    question TEXT,
    selected TEXT,
    correct TEXT,
    explanation TEXT,
    created_at TEXT
)
""")
conn.commit()

# =========================
# SAFE JSON EXTRACTION
# =========================

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except:
        return None

# =========================
# HARD QUESTION GENERATOR (FAIL-SAFE)
# =========================

def generate_question(context, retries=3):
    prompt = f"""
Create ONE HARD diagnostic MCQ like EI ASSET.

Context:
{context}

MANDATORY:
- Minimum TWO reasoning steps
- No direct formula substitution
- At least TWO misconception-based distractors
- All options must look plausible

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
        content = r.choices[0].message.content
        data = extract_json(content)
        if data and "question" in data:
            return data

    # absolute fallback (never crash)
    return {
        "question": "Diagnostic question generation failed. Please retry.",
        "options": ["A", "B", "C", "D"],
        "answer": "A",
        "explanation": "System fallback."
    }

# =========================
# UI – CLEAN EI-STYLE INDEX
# =========================

st.title("🧠 Diagnostic Assessment")
st.caption("EI-Style • Reasoning • Misconceptions")

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
        topic = "General Concept"

    num_q = st.slider("Number of Questions", 5, 15, 8)
    start = st.button("▶ Start Assessment")

# =========================
# GENERATE ASSESSMENT
# =========================

if start:
    st.session_state.questions = []
    st.session_state.responses = {}
    st.session_state.attempt_id = str(uuid.uuid4())

    context = f"""
Grade: {grade}
Subject: {subject}
Topic: {topic}
"""

    for _ in range(num_q):
        st.session_state.questions.append(
            generate_question(context)
        )

# =========================
# ASSESSMENT VIEW
# =========================

if "questions" in st.session_state:
    st.markdown("### Answer all questions. Answers appear only after submission.")

    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        st.session_state.responses[i] = st.radio(
            "",
            q["options"],
            key=f"q_{i}",
            index=None
        )
        st.markdown("---")

    submit = st.button("Submit Assessment")

    if submit:
        score = 0
        st.markdown("## 📊 Diagnostic Report")

        for i, q in enumerate(st.session_state.questions):
            selected = st.session_state.responses.get(i)
            correct = q["answer"]

            if selected == correct:
                score += 1

            cur.execute("""
            INSERT INTO attempts VALUES (?,?,?,?,?,?)
            """, (
                st.session_state.attempt_id,
                i + 1,
                q["question"],
                selected,
                correct,
                q["explanation"],
                datetime.now().isoformat()
            ))
            conn.commit()

            st.markdown(f"### Q{i+1}")
            st.write(q["question"])
            st.write(f"**Your Answer:** {selected}")
            st.write(f"**Correct Answer:** {correct}")
            st.write(f"**Explanation:** {q['explanation']}")
            st.markdown("---")

        st.markdown(f"## ✅ Score: {score} / {len(st.session_state.questions)}")
