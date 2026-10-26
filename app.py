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

def load_data():
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
def train_model(dataset_mtime):
    df = load_data()
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
    st.caption("Uploaded files are processed by this app while you use it. Avoid uploading sensitive personal information.")
    if DATA_PATH.exists():
        st.success("Resume.csv found beside app.py")
    else:
        st.warning("Copy Resume.csv beside app.py to enable category prediction.")

tab1, tab2, tab3 = st.tabs(["🔎 Analyze Resume", "📊 Model Evaluation", "ℹ️ About"])

with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("1. Upload resume")
        upload = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf","docx","txt"])
        resume = ""
        if upload:
            try:
                resume = read_resume(upload)
                if resume.strip():
                    st.success(f"Loaded {upload.name} · {len(resume.split())} words")
                else:
                    st.error("No selectable text found. Try another file or paste text below.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
        pasted = st.text_area("Or paste resume text", height=180, placeholder="Paste resume text here...")
        if pasted.strip():
            resume = pasted
    with c2:
        st.subheader("2. Job description")
        job = st.text_area("Paste target job description", height=275, placeholder="Paste the job description here...")
        st.caption("Similarity is a text-similarity estimate, not a hiring probability or validated ATS score.")
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
            if DATA_PATH.exists():
                with st.spinner("Training/loading the resume-category model. The first run may take a minute..."):
                    try:
                        result = train_model(DATA_PATH.stat().st_mtime)
                        if result:
                            vec, model, metrics, details = result
                            category = model.predict(vec.transform([clean_text(resume)]))[0]
                            st.success(f"Predicted resume category: **{category}**")
                            st.caption("The predicted category may be incorrect; review it manually.")
                        else:
                            st.warning("Could not train. Ensure Resume.csv contains Resume_str and Category columns.")
                    except Exception as e:
                        st.error(f"Model training failed: {e}")
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
    if DATA_PATH.exists() and st.button("Run evaluation for this app's Random Forest model"):
        with st.spinner("Training model and evaluating hold-out data..."):
            try:
                result = train_model(DATA_PATH.stat().st_mtime)
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
    st.markdown("1. Upload or paste resume text.\n2. Clean and preprocess the text.\n3. Extract skills using a transparent keyword list.\n4. Compare resume and job-description text using TF-IDF cosine similarity.\n5. Predict a resume category using a Random Forest trained on Resume.csv.")
    st.warning("This is a learning/demo tool, not a validated hiring system. Do not use its score or prediction as the sole basis for recruitment decisions.")
    st.markdown("**Technologies:** Python, Streamlit, pandas, scikit-learn, TF-IDF, Random Forest, cosine similarity, pypdf.")
