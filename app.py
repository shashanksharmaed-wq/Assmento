import streamlit as st
import openai
import sqlite3
import uuid
import json
import numpy as np
import pandas as pd
import faiss
from datetime import datetime

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="Academic Intelligence Platform", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

LLM_MODEL = "gpt-4.1"
EMBED_MODEL = "text-embedding-3-large"
SIM_THRESHOLD = 0.85

# =========================
# LOAD TSV (AUTO-DETECT)
# =========================

@st.cache_data
def load_syllabus():
    df = pd.read_csv("Teachshank_Master_Database_FINAL.tsv", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_syllabus()

# detect columns safely
CLASS_COL = [c for c in df.columns if c.lower() == "class"][0]
SUBJECT_COL = [c for c in df.columns if c.lower() == "subject"][0]
CHAPTER_COL = next((c for c in df.columns if c.lower() == "chapter"), None)
LO_COL = next((c for c in df.columns if c.lower() in ["lo", "learning_outcome", "learning outcome"]), None)

# =========================
# DATABASE
# =========================

conn = sqlite3.connect("questions.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    school_id TEXT,
    class INTEGER,
    subject TEXT,
    chapter TEXT,
    lo TEXT,
    question TEXT,
    options TEXT,
    answer TEXT,
    solution TEXT,
    embedding BLOB,
    created_at TEXT
)
""")
conn.commit()

# =========================
# EMBEDDING + FAISS
# =========================

def embed(text):
    r = openai.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding, dtype="float32")

_test = embed("init")
EMBED_DIM = _test.shape[0]
index = faiss.IndexFlatL2(EMBED_DIM)

def load_embeddings(school):
    cur.execute("SELECT embedding FROM questions WHERE school_id=?", (school,))
    for (e,) in cur.fetchall():
        v = np.frombuffer(e, dtype="float32")
        index.add(v.reshape(1, -1))

def is_unique(v):
    if index.ntotal == 0:
        return True
    D, _ = index.search(v.reshape(1, -1), 1)
    return D[0][0] > SIM_THRESHOLD

# =========================
# QUESTION GENERATION
# =========================

def generate_question(cls, subject, chapter, lo):
    prompt = f"""
Create ONE CBSE MCQ.

Class: {cls}
Subject: {subject}
Chapter: {chapter}
Learning Outcome: {lo}

Rules:
- NCERT aligned
- Conceptual
- Exam quality
- JSON only

JSON:
{{
 "question": "",
 "options": ["A","B","C","D"],
 "answer": "",
 "solution": ""
}}
"""
    r = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return json.loads(r.choices[0].message.content)

# =========================
# UI
# =========================

st.title("🧠 Academic Intelligence Platform")

with st.sidebar:
    school_id = st.text_input("School Code", "DEMO_SCHOOL")

    classes = sorted(df[CLASS_COL].unique())
    cls = st.selectbox("Class", classes)

    subjects = sorted(df[df[CLASS_COL] == cls][SUBJECT_COL].unique())
    subject = st.selectbox("Subject", subjects)

    if CHAPTER_COL:
        chapters = sorted(
            df[(df[CLASS_COL] == cls) & (df[SUBJECT_COL] == subject)][CHAPTER_COL].dropna().unique()
        )
        chapter = st.selectbox("Chapter", chapters)
    else:
        chapter = "N/A"

    if LO_COL:
        los = sorted(
            df[
                (df[CLASS_COL] == cls) &
                (df[SUBJECT_COL] == subject) &
                ((df[CHAPTER_COL] == chapter) if CHAPTER_COL else True)
            ][LO_COL].dropna().unique()
        )
        lo = st.selectbox("Learning Outcome", los)
    else:
        lo = "General Concept"

    generate_btn = st.button("Generate Question")

if "loaded" not in st.session_state:
    load_embeddings(school_id)
    st.session_state.loaded = True

# =========================
# PIPELINE
# =========================

if generate_btn:
    q = generate_question(cls, subject, chapter, lo)
    vec = embed(q["question"] + q["solution"])

    if not is_unique(vec):
        st.error("Similar question exists. Try again.")
    else:
        qid = str(uuid.uuid4())
        index.add(vec.reshape(1, -1))

        cur.execute("""
        INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            qid,
            school_id,
            cls,
            subject,
            chapter,
            lo,
            q["question"],
            json.dumps(q["options"]),
            q["answer"],
            q["solution"],
            vec.tobytes(),
            datetime.now().isoformat()
        ))
        conn.commit()

        st.success("Question Generated")

        st.subheader("Question")
        st.write(q["question"])

        for o in q["options"]:
            st.write(o)

        st.subheader("Answer")
        st.write(q["answer"])

        st.subheader("Solution")
        st.write(q["solution"])
