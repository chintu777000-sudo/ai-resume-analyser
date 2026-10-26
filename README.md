# AI Resume Analyzer

A Streamlit demo for resume text cleanup, skill keyword extraction, TF-IDF job-description similarity, and optional resume-category classification with Random Forest.

## Deploy on Streamlit Community Cloud

1. Put `app.py`, `requirements.txt`, and this README at the root of a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the main file.
3. Deploy. The app does not need secrets or API keys.

## Optional category classifier data

The category prediction and live evaluation buttons activate only when `Resume.csv` is placed beside `app.py`. The CSV must contain `Resume_str` and `Category` columns. This deployment package does not include that dataset. Do not add real people's resumes to a public repository; use only data you have permission to publish, or deploy without the category dataset. Resume upload, skill extraction, and job-description comparison work without it.

## Scope and privacy

The match percentage is a text-similarity estimate, not a hiring probability or validated ATS score. This is a learning/demo tool, not a hiring decision system. Uploaded resume content is processed by the hosted app while you use it; avoid uploading sensitive personal information.
