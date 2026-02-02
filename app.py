import streamlit as st
import openai
import pandas as pd
import numpy as np
import sqlite3
import json
import uuid
from datetime import datetime

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Assessment Intelligence (EI-style)", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

LLM_MODEL = "gpt-4.1"

# =========================
# LOAD SYLLABUS
# =========================

@st.cache_data
def load_syllabus():
    df = pd.read_csv("Teachshank_Master_Database_FINAL.tsv", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_syllabus()

def find_col(names):
    for c in df.columns:
        if c.lower().replace(" ", "_") in names:
            return c
    return None

CLASS_COL = find_col({"class", "grade", "std"})
SUBJECT_COL = find_col({"subject"})
CHAPTER_COL = find_col({"chapter", "unit"})
LO_COL = find_col({"lo", "learning_outcome", "learning outcome"})

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
# QUESTION GENERATOR (HARD)
# =========================

def generate_question(cls, subject, chapter, lo):
    prompt = f"""
You are designing a DIAGNOSTIC question like EI ASSET.

Class: {cls}
Subject: {subject}
Chapter: {chapter}
Learning Outcome: {lo}

MANDATORY RULES:
- Minimum TWO reasoning steps
- No direct formula substitution
- At least TWO options must be misconception-based
- All options must look plausible
- Difficulty should challenge top 25% students

Return JSON ONLY:
{{
 "question": "",
 "options": ["A","B","C","D"],
 "answer": "",
 "explanation": "Explain why each wrong option is tempting but incorrect"
}}
"""
    r = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    return json.loads(r.choices[0].message.content)

# =========================
# UI – EI ASSET STYLE
# =========================

st.title("🧠 Diagnostic Assessment (EI-Style)")
st.caption("Thinking • Reasoning • Misconception Detection")

with st.sidebar:
    cls = st.selectbox("Class", sorted(df[CLASS_COL].unique()))
    subject = st.selectbox(
        "Subject",
        sorted(df[df[CLASS_COL] == cls][SUBJECT_COL].unique())
    )

    chapters = sorted(
        df[(df[CLASS_COL] == cls) & (df[SUBJECT_COL] == subject)][CHAPTER_COL].unique()
    )
    chapter = st.selectbox("Chapter", chapters)

    los = sorted(
        df[
            (df[CLASS_COL] == cls) &
            (df[SUBJECT_COL] == subject) &
            (df[CHAPTER_COL] == chapter)
        ][LO_COL].unique()
    )
    lo = st.selectbox("Learning Outcome", los)

    num_q = st.slider("Number of Questions", 5, 15, 8)
    start = st.button("Start Assessment")

# =========================
# GENERATE FULL TEST
# =========================

if start:
    st.session_state.attempt_id = str(uuid.uuid4())
    st.session_state.questions = []
    st.session_state.responses = {}

    for _ in range(num_q):
        q = generate_question(cls, subject, chapter, lo)
        st.session_state.questions.append(q)

# =========================
# ASSESSMENT VIEW
# =========================

if "questions" in st.session_state:

    st.markdown("### Answer ALL questions. Results appear only after submission.")

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

    # =========================
    # RESULTS (POST-SUBMIT)
    # =========================

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

        st.markdown("""
        **Interpretation (EI-style):**
        - High score → Strong conceptual clarity
        - Medium score → Partial misconceptions
        - Low score → Foundational gaps detected
        """)
