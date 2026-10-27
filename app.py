import re
from pathlib import Path
import io
import warnings

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "Resume.csv"

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.4rem; max-width:1200px;}
.hero {padding:1.5rem 1.7rem;border-radius:16px;background:linear-gradient(120deg,#102a43,#176b87);color:white;margin-bottom:1rem}
.hero h1 {color:white;margin:0}.hero p {color:#e4f4fa;margin:.4rem 0 0}
div[data-testid="stMetric"] {background:white;border:1px solid #e2e8f0;padding:12px;border-radius:12px}
</style>
<div class="hero"><h1>📄 AI Resume Analyzer</h1>
<p>Resume classification, job-description matching, and skill-gap insights powered by machine learning.</p></div>
""", unsafe_allow_html=True)

def clean_text(value):
    value = str(value or "").lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9+#.\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

SKILLS = ["python","java","c++","sql","machine learning","deep learning",
"natural language processing","nlp","tensorflow","pytorch","scikit-learn",
"pandas","numpy","data analysis","data visualization","power bi","tableau",
"excel","aws","azure","docker","kubernetes","git","github","html","css",
"javascript","react","node.js","flask","fastapi","django","spring boot",
"mysql","mongodb","communication","leadership","problem solving","statistics",
"computer vision","transformers","bert","rest api","linux"]

# Built-in fictional demo data so the app can be demonstrated without uploading files.
SAMPLE_RESUMES = {
    "Alex Morgan — Entry-level ML Engineer": """Alex Morgan
Hyderabad, India | alex.morgan@example.com | +91-XXXXXXXXXX

SUMMARY
Entry-level AI and Machine Learning graduate with hands-on experience in Python, data preprocessing, supervised learning, and NLP. Interested in building reliable data-driven applications.

EDUCATION
B.Tech in Computer Science (AI & ML) | 2024–2028

TECHNICAL SKILLS
Python, SQL, Machine Learning, Natural Language Processing, Pandas, NumPy, scikit-learn, TensorFlow, Git, Data Analysis, Statistics, REST API

PROJECTS
AI Resume Analyzer: Cleaned resume text, built TF-IDF features, compared Logistic Regression, Random Forest, and Linear SVM, and evaluated classification performance using accuracy and macro F1.
Student Score Predictor: Prepared tabular data, trained a regression model, and evaluated predictions using MAE and R-squared.

EXPERIENCE
Academic ML Projects | 2025–2026
Cleaned datasets, documented experiments, visualized results, and explained model limitations.

COURSEWORK
Data Structures, Database Systems, Probability and Statistics, Machine Learning
""",
    "Priya Sharma — Python Developer": """Priya Sharma
Hyderabad, India | priya.sharma@example.com | +91-XXXXXXXXXX

SUMMARY
Computer Science student focused on Python development, backend APIs, SQL databases, testing, and clean software design.

TECHNICAL SKILLS
Python, Java, SQL, FastAPI, Flask, REST API, MySQL, Git, GitHub, HTML, CSS, JavaScript, Docker, Linux, Problem Solving

PROJECTS
Task Tracker API: Built CRUD endpoints with FastAPI, validated request data, and connected the service to a relational database.
Student Score Predictor: Created a Python application that accepts input features and returns model predictions.

EDUCATION
B.Tech in Computer Science | 2024–2028

EXPERIENCE
Student Developer | 2025–2026
Worked on course projects, debugging, version control, and API documentation.
""",
    "Rahul Verma — Data Analyst": """Rahul Verma
Hyderabad, India | rahul.verma@example.com | +91-XXXXXXXXXX

SUMMARY
Aspiring Data Analyst with experience in cleaning datasets, SQL querying, exploratory analysis, dashboard preparation, and communicating insights.

TECHNICAL SKILLS
SQL, Python, Pandas, NumPy, Excel, Power BI, Tableau, Data Analysis, Data Visualization, Statistics, MySQL, Communication

PROJECTS
Sales Analysis Dashboard: Cleaned sales records, analyzed monthly revenue and product trends, and summarized findings with charts.
Student Performance Analysis: Used Python and Pandas to explore score distributions and identify patterns in academic data.

EDUCATION
B.Tech in Computer Science | 2024–2028

EXPERIENCE
Academic Data Projects | 2025–2026
Prepared reports, validated data quality, and presented findings to classmates.
"""
}

SAMPLE_JOBS = {
    "Machine Learning Engineer (Entry Level)": """Role: Entry-Level Machine Learning Engineer

Required skills: Python, machine learning, data preprocessing, Pandas, NumPy, scikit-learn, statistics, model evaluation, SQL, and Git.

Responsibilities: Prepare datasets, train and evaluate classification or regression models, document experiments, and collaborate with software and data teams. Familiarity with NLP or deep learning is a plus.
""",
    "Python Backend Developer": """Role: Junior Python Backend Developer

Required skills: Python, FastAPI or Flask, REST APIs, SQL, MySQL, Git, debugging, testing, and basic Linux knowledge.

Responsibilities: Build and maintain backend endpoints, validate data, integrate databases, write readable code, fix defects, and document API behavior. Docker experience is a plus.
""",
    "Data Analyst": """Role: Junior Data Analyst

Required skills: SQL, Python, Pandas, Excel, statistics, data cleaning, data analysis, and data visualization. Power BI or Tableau experience is preferred.

Responsibilities: Clean and validate datasets, write queries, identify trends, create dashboards, and communicate actionable insights to stakeholders.
"""
}

def apply_resume_sample():
    choice = st.session_state.get("resume_sample_choice", "Custom / upload my own")
    st.session_state["resume_text_input"] = SAMPLE_RESUMES.get(choice, "")
    if choice in SAMPLE_JOBS:
        st.session_state["job_sample_choice"] = choice
        st.session_state["job_text_input"] = SAMPLE_JOBS[choice]

def apply_job_sample():
    choice = st.session_state.get("job_sample_choice", "Custom job description")
    st.session_state["job_text_input"] = SAMPLE_JOBS.get(choice, "")

def extract_skills(text):
    low = clean_text(text)
    return sorted({s.title() for s in SKILLS if s in low}, key=str.lower)

def read_resume(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.getvalue()
    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def load_data(source_bytes=None):
    if source_bytes is not None:
        df = pd.read_csv(io.BytesIO(source_bytes))
    else:
        if not DATA_PATH.exists():
            return None
        df = pd.read_csv(DATA_PATH)
    if "Resume_str" not in df.columns or "Category" not in df.columns:
        return None
    df = df[["Resume_str","Category"]].dropna().copy()
    df["Resume_str"] = df["Resume_str"].astype(str).map(clean_text)
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[(df["Resume_str"].str.len() > 30) & (df["Category"] != "")]
    return df.drop_duplicates(subset=["Resume_str"]).reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def train_model(dataset_bytes):
    df = load_data(dataset_bytes)
    if df is None or len(df) < 20 or df["Category"].nunique() < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        df["Resume_str"], df["Category"], test_size=.2, random_state=42, stratify=df["Category"])
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=.95, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1,
                                   class_weight="balanced_subsample")
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    metrics = {"accuracy": accuracy_score(y_test, pred),
               "macro_f1": f1_score(y_test, pred, average="macro")}
    return vec, model, metrics, (len(df), len(X_train), len(X_test), df["Category"].nunique())

with st.sidebar:
    st.header("Project information")
    st.write("**Project:** AI Resume Analyzer")
    st.write("**Methods:** TF-IDF, Random Forest, cosine similarity")
    st.caption("Files are processed locally by this app.")
    if DATA_PATH.exists():
        st.success("Resume.csv found beside app.py")
    else:
        st.info("Upload Resume.csv below to enable resume-category prediction.")
    dataset_upload = st.file_uploader("Optional training dataset (Resume.csv)", type=["csv"], help="Must contain Resume_str and Category columns. The dataset is used in memory for this session.")
    dataset_bytes = dataset_upload.getvalue() if dataset_upload else None
    if dataset_bytes is None and DATA_PATH.exists():
        dataset_bytes = DATA_PATH.read_bytes()

tab1, tab2, tab3 = st.tabs(["🔎 Analyze Resume", "📊 Model Evaluation", "ℹ️ About"])

with tab1:
    st.markdown("### Quick demo mode")
    st.caption("Choose a fictional sample resume and a sample job description to populate the fields instantly. You can edit either field or upload your own resume.")
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.selectbox("Sample resume", ["Custom / upload my own", *SAMPLE_RESUMES.keys()], key="resume_sample_choice", on_change=apply_resume_sample)
    with d2:
        st.selectbox("Sample job description", ["Custom job description", *SAMPLE_JOBS.keys()], key="job_sample_choice", on_change=apply_job_sample)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("1. Resume")
        upload = st.file_uploader("Upload resume (PDF, DOCX, or TXT)", type=["pdf","docx","txt"], key="resume_upload")
        uploaded_resume_text = ""
        if upload:
            try:
                uploaded_resume_text = read_resume(upload)
                if uploaded_resume_text.strip():
                    st.success(f"Loaded {upload.name} · {len(uploaded_resume_text.split())} words")
                else:
                    st.error("No selectable text found. Try another file or paste text below.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
        if "resume_text_input" not in st.session_state:
            st.session_state["resume_text_input"] = ""
        pasted = st.text_area("Or use/edit resume text", height=275, key="resume_text_input", placeholder="Choose a sample resume above, upload a file, or paste resume text here...")
        resume = uploaded_resume_text if upload and uploaded_resume_text.strip() else pasted
        if upload and uploaded_resume_text.strip():
            st.caption("The uploaded file is used for analysis while it is selected. Clear the upload to use the text box instead.")
    with c2:
        st.subheader("2. Job description")
        if "job_text_input" not in st.session_state:
            st.session_state["job_text_input"] = ""
        job = st.text_area("Use/edit job description", height=275, key="job_text_input", placeholder="Choose a sample job description above or paste one here...")
        st.caption("Text similarity is an estimate of textual overlap, not a hiring probability or validated ATS score.")

    if st.button("Analyze Resume", type="primary", use_container_width=True):
        if len(clean_text(resume)) < 30:
            st.error("Upload a readable resume or paste resume text first.")
        else:
            st.subheader("3. Analysis results")
            rskills = extract_skills(resume)
            if job.strip():
                vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=12000)
                mat = vec.fit_transform([clean_text(resume), clean_text(job)])
                score = cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100
                jskills = extract_skills(job)
                matched = sorted(set(rskills) & set(jskills), key=str.lower)
                missing = sorted(set(jskills) - set(rskills), key=str.lower)
                a,b,c = st.columns(3)
                a.metric("Text similarity", f"{score:.1f}%")
                b.metric("Resume skills detected", len(rskills))
                c.metric("Job skills detected", len(jskills))
                st.progress(float(min(max(score/100,0),1)))
                left,right = st.columns(2)
                with left:
                    st.markdown("**Skills detected in resume**")
                    st.write(", ".join(rskills) if rskills else "No listed skills detected.")
                    st.markdown("**Skills present in both**")
                    st.write(", ".join(matched) if matched else "No overlap detected.")
                with right:
                    st.markdown("**Skills detected in job description**")
                    st.write(", ".join(jskills) if jskills else "No listed skills detected.")
                    st.markdown("**Job skills not found in resume**")
                    st.write(", ".join(missing) if missing else "No missing skills detected.")
                if missing:
                    st.info("Only add missing skills to your resume if you genuinely have that experience.")
            else:
                st.metric("Resume word count", len(resume.split()))
                st.write("**Skills detected:**", ", ".join(rskills) if rskills else "No listed skills detected.")
                st.info("Add a job description to compare text and skills.")
            if dataset_bytes is not None:
                with st.spinner("Training/loading the resume-category model. The first run may take a minute..."):
                    try:
                        result = train_model(dataset_bytes)
                        if result:
                            vec, model, metrics, details = result
                            category = model.predict(vec.transform([clean_text(resume)]))[0]
                            st.success(f"Predicted resume category: **{category}**")
                            st.caption("The predicted category may be incorrect; review it manually.")
                        else:
                            st.warning("Could not train. Ensure Resume.csv contains Resume_str and Category columns.")
                    except Exception as e:
                        st.error(f"Model training failed: {e}")
            else:
                st.info("Category prediction is optional and requires Resume.csv (or a CSV uploaded in the sidebar). Resume/job matching works without it.")
            with st.expander("View extracted resume text"):
                st.text(resume[:12000])

with tab2:
    st.subheader("Model evaluation")
    st.write("These are the previously recorded notebook experiment results; they are not freshly computed by this page.")
    results = pd.DataFrame([
        ["Logistic Regression",63.98,0.5964],
        ["Random Forest",73.44,0.6865],
        ["Linear SVM",70.22,0.6651],
        ["Tuned Logistic Regression",66.60,0.6269],
        ["Tuned SVM",68.01,0.6402],
    ], columns=["Model","Accuracy (%)","Macro F1"])
    st.dataframe(results, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(9,4))
    bars = ax.bar(results["Model"], results["Accuracy (%)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0,100)
    ax.set_title("Previously Recorded Model Accuracy")
    ax.tick_params(axis="x", rotation=18)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    if dataset_bytes is not None and st.button("Run evaluation for this app's Random Forest model"):
        with st.spinner("Training model and evaluating hold-out data..."):
            try:
                result = train_model(dataset_bytes)
                if result:
                    _,_,metrics,details = result
                    a,b = st.columns(2)
                    a.metric("App model accuracy", f"{metrics['accuracy']*100:.2f}%")
                    b.metric("App model macro F1", f"{metrics['macro_f1']:.4f}")
                    st.write(f"Clean records: {details[0]} · Training: {details[1]} · Testing: {details[2]} · Categories: {details[3]}")
                    st.caption("This app trains a separate model; scores can differ from notebook results.")
                else:
                    st.error("Could not train model. Verify dataset columns and categories.")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

with tab3:
    st.subheader("Project overview")
    st.write("The AI Resume Analyzer demonstrates resume text preprocessing, TF-IDF feature extraction, Random Forest resume-category classification, and job-description matching using cosine similarity.")
    st.markdown("**Workflow**")
    st.markdown("1. Upload or paste resume text.\n2. Clean and preprocess the text.\n3. Extract skills using a transparent keyword list.\n4. Compare resume and job-description text using TF-IDF cosine similarity.\n5. Optionally predict a resume category using a Random Forest trained on Resume.csv.")
    st.warning("This is a learning/demo tool, not a validated hiring system. Do not use its score or prediction as the sole basis for recruitment decisions.")
    st.markdown("**Technologies:** Python, Streamlit, pandas, scikit-learn, TF-IDF, Random Forest, cosine similarity, pypdf.")
