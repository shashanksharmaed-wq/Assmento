import streamlit as st
import openai
import pandas as pd
import sqlite3
import json
import uuid
from datetime import datetime

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="EI-Style Diagnostic Assessment", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]
LLM_MODEL = "gpt-4.1"

# =========================
# LOAD SYLLABUS SAFELY
# =========================

@st.cache_data
def load_syllabus():
    df = pd.read_csv("Teachshank_Master_Database_FINAL.tsv", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_syllabus()

def find_col(candidates):
    for col in df.columns:
        key = col.lower().replace(" ", "_")
        if key in candidates:
            return col
    return None

CLASS_COL = find_col({"class", "grade", "std", "class_name"})
SUBJECT_COL = find_col({"subject", "sub", "subject_name"})
CHAPTER_COL = find_col({"chapter", "unit", "lesson"})
LO_COL = find_col({"lo", "learning_outcome", "learning outcome", "objective"})

if not CLASS_COL:
    st.error("❌ Class column not found in syllabus file")
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
# HARD QUESTION GENERATOR
# =========================

def generate_question(cls, subject, chapter, lo):
    prompt = f"""
Create ONE HARD diagnostic MCQ like EI ASSET.

Class: {cls}
Subject: {subject}
Chapter: {chapter}
Learning Outcome: {lo}

Rules:
- At least 2 reasoning steps
- No direct formula substitution
- 2 misconception-based distractors
- All options plausible

Return JSON ONLY:
{{
 "question": "",
 "options": ["A","B","C","D"],
 "answer": "",
 "explanation": "Explain why each wrong option is tempting"
}}
"""
    r = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    return json.loads(r.choices[0].message.content)

# =========================
# UI (EI ASSET STYLE)
# =========================

st.title("🧠 Diagnostic Assessment (EI-Style)")
st.caption("Reasoning • Misconceptions • Thinking Skills")

with st.sidebar:
    classes = sorted(df[CLASS_COL].dropna().unique())
    cls = st.selectbox("Class", classes)

    if SUBJECT_COL:
        subjects = sorted(df[df[CLASS_COL] == cls][SUBJECT_COL].dropna().unique())
        subject = st.selectbox("Subject", subjects)
    else:
        subject = "General"

    if CHAPTER_COL:
        chapters = sorted(
            df[
                (df[CLASS_COL] == cls) &
                ((df[SUBJECT_COL] == subject) if SUBJECT_COL else True)
            ][CHAPTER_COL].dropna().unique()
        )
        chapter = st.selectbox("Chapter", chapters)
    else:
        chapter = "General"

    if LO_COL:
        los = sorted(
            df[
                (df[CLASS_COL] == cls) &
                ((df[SUBJECT_COL] == subject) if SUBJECT_COL else True) &
                ((df[CHAPTER_COL] == chapter) if CHAPTER_COL else True)
            ][LO_COL].dropna().unique()
        )
        lo = st.selectbox("Learning Outcome", los)
    else:
        lo = "Conceptual Understanding"

    num_q = st.slider("Number of Questions", 5, 15, 8)
    start = st.button("Start Assessment")

# =========================
# GENERATE TEST
# =========================

if start:
    st.session_state.questions = []
    st.session_state.responses = {}
    st.session_state.attempt_id = str(uuid.uuid4())

    for _ in range(num_q):
        q = generate_question(cls, subject, chapter, lo)
        st.session_state.questions.append(q)

# =========================
# ASSESSMENT VIEW
# =========================

if "questions" in st.session_state:
    st.markdown("### Answer all questions. Results shown after submission.")

    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}. {q['question']}**")
        choice = st.radio(
            "",
            q["options"],
            key=f"q_{i}",
            index=None
        )
        st.session_state.responses[i] = choice
        st.markdown("---")

    submit = st.button("Submit Assessment")

    if submit:
        score = 0
        st.markdown("## 📊 Diagnostic Report")

        for i, q in enumerate(st.session_state.questions):
            sel = st.session_state.responses.get(i)
            if sel == q["answer"]:
                score += 1

            cur.execute("""
            INSERT INTO attempts VALUES (?,?,?,?,?,?)
            """, (
                st.session_state.attempt_id,
                i + 1,
                q["question"],
                sel,
                q["answer"],
                q["explanation"],
                datetime.now().isoformat()
            ))
            conn.commit()

            st.markdown(f"### Q{i+1}")
            st.write(q["question"])
            st.write(f"Your Answer: {sel}")
            st.write(f"Correct Answer: {q['answer']}")
            st.write(f"Explanation: {q['explanation']}")
            st.markdown("---")

        st.markdown(f"## ✅ Score: {score} / {len(st.session_state.questions)}")
