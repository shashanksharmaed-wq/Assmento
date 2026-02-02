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
# LOAD TSV (NO ASSUMPTIONS)
# =========================

@st.cache_data
def load_tsv():
    df = pd.read_csv("Teachshank_Master_Database_FINAL.tsv", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_tsv()

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
# QUESTION GENERATOR (EI HARD)
# =========================

def generate_question(context):
    prompt = f"""
Create ONE HARD diagnostic MCQ like EI ASSET.

Context:
{context}

MANDATORY:
- Minimum 2 reasoning steps
- No direct formula substitution
- At least 2 misconception-based distractors
- All options must look plausible

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
# UI – COLUMN MAPPING (FAIL-SAFE)
# =========================

st.title("🧠 Diagnostic Assessment (EI-Style)")
st.caption("Thinking • Reasoning • Misconception Detection")

st.markdown("### Syllabus Mapping (auto-safe)")

with st.expander("📄 TSV Columns Detected"):
    st.write(list(df.columns))

with st.sidebar:
    class_col = st.selectbox("Select Class Column", df.columns)
    subject_col = st.selectbox("Select Subject Column", df.columns)
    chapter_col = st.selectbox("Select Chapter Column", df.columns)
    lo_col = st.selectbox("Select LO Column (optional)", ["None"] + list(df.columns))

    classes = sorted(df[class_col].dropna().unique())
    cls = st.selectbox("Class", classes)

    subjects = sorted(df[df[class_col] == cls][subject_col].dropna().unique())
    subject = st.selectbox("Subject", subjects)

    chapters = sorted(
        df[(df[class_col] == cls) & (df[subject_col] == subject)][chapter_col]
        .dropna().unique()
    )
    chapter = st.selectbox("Chapter", chapters)

    if lo_col != "None":
        los = sorted(
            df[
                (df[class_col] == cls) &
                (df[subject_col] == subject) &
                (df[chapter_col] == chapter)
            ][lo_col].dropna().unique()
        )
        lo = st.selectbox("Learning Outcome", los)
    else:
        lo = "Conceptual Reasoning"

    num_q = st.slider("Number of Questions", 5, 15, 8)
    start = st.button("Start Assessment")

# =========================
# GENERATE TEST
# =========================

if start:
    st.session_state.questions = []
    st.session_state.responses = {}
    st.session_state.attempt_id = str(uuid.uuid4())

    context = f"""
Class: {cls}
Subject: {subject}
Chapter: {chapter}
Learning Outcome: {lo}
"""

    for _ in range(num_q):
        q = generate_question(context)
        st.session_state.questions.append(q)

# =========================
# ASSESSMENT VIEW (EI STYLE)
# =========================

if "questions" in st.session_state:

    st.markdown("### Answer all questions. Answers shown only after submission.")

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
            st.write(f"Your Answer: {selected}")
            st.write(f"Correct Answer: {correct}")
            st.write(f"Explanation: {q['explanation']}")
            st.markdown("---")

        st.markdown(f"## ✅ Score: {score} / {len(st.session_state.questions)}")
        st.markdown("""
**EI-style interpretation**
- High score → Strong conceptual clarity
- Medium score → Partial misconceptions
- Low score → Foundational gaps detected
""")
