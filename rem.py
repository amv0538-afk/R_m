import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer, util

# ===============================
# 🧩 Load the model
# ===============================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ===============================
# 📄 Extract text from PDF
# ===============================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

# ===============================
# 🎯 Check if required skills exist
# ===============================
def has_required_skills(resume_text, required_skills):
    for skill in required_skills:
        if skill.lower() not in resume_text:
            return False
    return True

# ===============================
# 🔍 Compute similarity ranking
# ===============================
def get_similarity(resume_texts, required_skills):
    skill_sentence = " ".join(required_skills)
    skill_embedding = model.encode(skill_sentence, convert_to_tensor=True)
    results = []

    for name, text in resume_texts.items():
        if has_required_skills(text, required_skills):
            resume_embedding = model.encode(text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(skill_embedding, resume_embedding).item()
            results.append((name, similarity))

    return sorted(results, key=lambda x: x[1], reverse=True)

# ===============================
# 🧠 Streamlit App UI
# ===============================
st.set_page_config(page_title="AI Resume Screener", page_icon="🤖", layout="centered")
st.title("🤖 AI Resume Screener for Recruiters & Students")
st.write("Upload resumes and filter candidates by required skills.")

uploaded_files = st.file_uploader("📂 Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])
skills = st.text_input("🧠 Enter Required Skills (comma-separated)", "Python, SQL, Machine Learning")

if uploaded_files and skills:
    required_skills = [s.strip() for s in skills.split(",")]

    with st.spinner("Analyzing resumes..."):
        resume_texts = {file.name: extract_text_from_pdf(file) for file in uploaded_files}
        results = get_similarity(resume_texts, required_skills)

    st.subheader("✅ Shortlisted Candidates")
    if results:
        df = pd.DataFrame(results, columns=["Resume", "Match_Score"])
        df["Match_Score"] = (df["Match_Score"] * 100).round(2)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results (CSV)", csv, "shortlisted_resumes.csv", "text/csv")
    else:
        st.warning("No resumes matched all required skills.")
