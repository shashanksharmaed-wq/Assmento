# =====================================================================
# ACADEMIC INTELLIGENCE PLATFORM – FULL PRODUCT
# CBSE Class 9–12 | Assessment | Practice | Exams
# Teacher-Proof | Sales-Demo Ready | Investor-Presentable
# =====================================================================

import streamlit as st
import openai
import sqlite3
import uuid
import json
import numpy as np
import faiss
import matplotlib.pyplot as plt
from datetime import datetime
import random

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Academic Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

openai.api_key = st.secrets["OPENAI_API_KEY"]

LLM_MODEL = "gpt-4.1"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536
SIM_THRESHOLD = 0.85

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
    question TEXT,
    options TEXT,
    answer TEXT,
    solution TEXT,
    bloom TEXT,
    steps INTEGER,
    load TEXT,
    traps INTEGER,
    diagram_spec TEXT,
    embedding BLOB,
    created_at TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS exam_papers (
    id TEXT PRIMARY KEY,
    school_id TEXT,
    class INTEGER,
    subject TEXT,
    paper TEXT,
    created_at TEXT
)
""")

conn.commit()

# =========================
# FAISS (ANTI-REPEAT)
# =========================

index = faiss.IndexFlatL2(EMBED_DIM)

def load_school_embeddings(school_id):
    cur.execute("SELECT embedding FROM questions WHERE school_id=?", (school_id,))
    for (emb,) in cur.fetchall():
        vec = np.frombuffer(emb, dtype="float32")
        index.add(vec.reshape(1, -1))

# =========================
# EMBEDDING
# =========================

def embed(text):
    r = openai.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding).astype("float32")

def unique(vec):
    if index.ntotal == 0:
        return True
    D, _ = index.search(vec.reshape(1, -1), 1)
    return D[0][0] > SIM_THRESHOLD

# =========================
# DIAGRAM ENGINE
# =========================

def render_diagram(spec):
    t = spec["type"]

    if t == "triangle":
        pts = spec["points"]
        A, B, C = pts["A"], pts["B"], pts["C"]
        x = [A[0], B[0], C[0], A[0]]
        y = [A[1], B[1], C[1], A[1]]
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.text(*A, "A"); plt.text(*B, "B"); plt.text(*C, "C")
        plt.axis("equal"); plt.axis("off")
        st.pyplot(plt.gcf()); plt.clf()

    elif t == "graph":
        x = np.linspace(-10, 10, 400)
        y = eval(spec["expression"])
        plt.figure()
        plt.plot(x, y)
        plt.axhline(0); plt.axvline(0); plt.grid()
        st.pyplot(plt.gcf()); plt.clf()

    elif t == "ray":
        plt.figure()
        plt.plot([-10, 10], [0, 0])
        plt.plot([0, 0], [-5, 5])
        ox, oy = spec["object"]
        ix, iy = spec["image"]
        plt.arrow(ox, 0, 0, oy, head_width=0.2)
        plt.arrow(ix, 0, 0, iy, head_width=0.2, color="red")
        plt.axis("off")
        st.pyplot(plt.gcf()); plt.clf()

# =========================
# QUESTION GENERATION
# =========================

def generate_question(payload):
    prompt = f"""
Create ONE elite CBSE MCQ.

Class {payload['class']} | {payload['subject']} | {payload['chapter']}
Bloom: {payload['bloom']}
Steps: {payload['steps']}
Cognitive Load: {payload['load']}
Trap Density: {payload['traps']}

Rules:
- NCERT aligned
- No recall if Bloom != Remember
- Exam-quality language
- If diagram needed, return diagram_spec JSON

JSON:
{{
 "question": "",
 "options": ["A","B","C","D"],
 "answer": "",
 "solution": "",
 "diagram_spec": null OR {{...}}
}}
"""
    r = openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return json.loads(r.choices[0].message.content)

# =========================
# EXAM PAPER GENERATOR
# =========================

def generate_exam_paper(school_id, cls, subject, count=20):
    cur.execute("""
    SELECT question, options, answer
    FROM questions
    WHERE school_id=? AND class=? AND subject=?
    ORDER BY RANDOM() LIMIT ?
    """, (school_id, cls, subject, count))

    qs = cur.fetchall()
    paper = []
    for i, q in enumerate(qs, 1):
        paper.append({
            "qno": i,
            "question": q[0],
            "options": json.loads(q[1]),
            "answer": q[2]
        })

    pid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO exam_papers VALUES (?,?,?,?,?,?)",
        (pid, school_id, cls, subject, json.dumps(paper), datetime.now().isoformat())
    )
    conn.commit()
    return paper

# =========================
# UI – SALES DEMO HEADER
# =========================

st.markdown("""
# 🧠 Academic Intelligence Platform
### *Assessment • Practice • Exams (CBSE 9–12)*

**What schools get**
- Zero question repetition
- Diagram-based reasoning
- Exam-ready papers
- School-isolated repositories
""")

# =========================
# SIDEBAR – TEACHER SAFE
# =========================

with st.sidebar:
    school_id = st.text_input("School Code", "DEMO_SCHOOL")
    cls = st.selectbox("Class", [9,10,11,12])
    subject = st.selectbox("Subject", ["Maths","Physics","Chemistry","Biology"])
    chapter = st.text_input("Chapter")
    mode = st.radio("Mode", ["Generate Question", "Create Exam Paper"])
    bloom = st.selectbox("Bloom Level", ["Remember","Apply","Analyze","Evaluate"])
    steps = st.slider("Step Depth", 1, 5)
    load = st.selectbox("Cognitive Load", ["Low","Medium","High"])
    traps = st.slider("Trap Density", 0, 3)
    action = st.button("Run")

if "loaded" not in st.session_state:
    load_school_embeddings(school_id)
    st.session_state.loaded = True

# =========================
# PIPELINE
# =========================

if action and mode == "Generate Question":
    payload = {
        "class": cls,
        "subject": subject,
        "chapter": chapter,
        "bloom": bloom,
        "steps": steps,
        "load": load,
        "traps": traps
    }

    q = generate_question(payload)
    vec = embed(q["question"] + q["solution"])

    if unique(vec):
        qid = str(uuid.uuid4())
        index.add(vec.reshape(1, -1))

        cur.execute("""
        INSERT INTO questions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            qid, school_id, cls, subject, chapter,
            q["question"], json.dumps(q["options"]),
            q["answer"], q["solution"],
            bloom, steps, load, traps,
            json.dumps(q["diagram_spec"]),
            vec.tobytes(),
            datetime.now().isoformat()
        ))
        conn.commit()

        st.success("Question Locked & Saved")

        st.write("### Question")
        st.write(q["question"])

        if q["diagram_spec"]:
            render_diagram(q["diagram_spec"])

        for opt in q["options"]:
            st.write(opt)

        st.write("**Answer:**", q["answer"])

elif action and mode == "Create Exam Paper":
    paper = generate_exam_paper(school_id, cls, subject)

    st.success("Exam Paper Generated")
    for q in paper:
        st.markdown(f"**Q{q['qno']}. {q['question']}**")
        for o in q["options"]:
            st.write(o)
        st.markdown("---")
