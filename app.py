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

st.set_page_config(
    page_title="Diagnostic Assessment (EI-Style)",
    layout="wide",
    initial_sidebar_state="expanded"
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
LLM_MODEL = "gpt-4.1"

# =========================
# LOAD SYLLABUS (SAFE)
# =========================

@st.cache_data
def load_syllabus():
    df = pd.read_csv("Teachshank_Master_Database_FINAL.tsv", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_syllabus()

# --- auto-detect columns once ---
def find_col(possible):
    for c in df.columns:
        key = c.lower().replace(" ", "_")
        if key in possible:
            return c
    return None

CLASS_COL   = find_col({"class", "grade", "std"})
SUBJECT_COL = find_col({"subject", "subject_name", "sub"})
CHAPTER_COL = find_col({"chapter", "unit", "lesson"})
LO_COL      = find_col({"lo", "learning_outcome", "learning outcome"})

if not CLASS_COL or not SUBJECT_COL or not CHAPTER_COL:
    st.error("Required syllabus columns (Class / Subject / Chapter) not found.")
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

def generate_question(context):
    prompt = f"""
Create ONE HARD diagnostic MCQ like EI ASSET.

Context:
{context}

MANDATORY RULES:
- At least TWO reasoning steps
- No direct formula substitution
- Minimum TWO misconception-based distractors
- All options must look plausible

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
# UI – CLEAN EI-STYLE INDEX
# =========================

st.title("🧠 Diagnostic Assessment")
st.caption("EI-Style • Thinking • Reasoning • Misconceptions")

with st.sidebar:
    st.markdown("### 📘 Assessment Setup")

    # 1️⃣ GRADE
    grades = sorted(df[CLASS_COL].dropna().unique())
    grade = st.selectbox("Grade", grades)

    # 2️⃣ SUBJECT
    subjects = sorted(
        df[df[CLASS_COL] == grade][SUBJECT_COL].dropna().unique()
    )
    subject = st.selectbox("Subject", subjects)

    # 3️⃣ CHAPTER
    chapters = sorted(
        df[
            (df[CLASS_COL] == grade) &
            (df[SUBJECT_COL] == subject)
        ][CHAPTER_COL].dropna().unique()
    )
    chapter = st.selectbox("Chapter", chapters)

    # 4️⃣ LEARNING OUTCOME (OPTIONAL)
    if LO_COL:
        los = sorted(
            df[
                (df[CLASS_COL] == grade) &
                (df[SUBJECT_COL] == subject) &
                (df[CHAPTER_COL] == chapter)
            ][LO_COL].dropna().unique()
        )
        lo = st.selectbox("Learning Outcome", ["All"] + los)
    else:
        lo = "Conceptual Understanding"

    st.markdown("---")
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
Chapter: {chapter}
Learning Outcome: {lo}
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

    # =========================
    # RESULTS
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
**Interpretation (EI-Style)**  
• High score → Strong conceptual clarity  
• Medium score → Partial misconceptions  
• Low score → Foundational gaps detected  
""")
