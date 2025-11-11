import streamlit as st
import requests
from openai import OpenAI

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="🎯 Radhe Intern Finder", layout="wide")

st.title("🎯 Radhe Intern Finder")
st.caption("Find and summarize internships that fit your interests — powered by SerpAPI & GPT.")

# Load secrets safely
serpapi_key = st.secrets["serpapi"]["key"]
openai_key = st.secrets.get("openai", {}).get("key")

client = None
if openai_key:
    client = OpenAI(api_key=openai_key)

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("🔍 Search Settings")
    query = st.text_input("Keywords", "finance intern, investment, analyst")
    location = st.text_input("Location", "Atlanta, GA")
    num_results = st.slider("Number of results", 5, 30, 10)
    st.markdown("---")
    st.write("💡 Tip: Try `data science`, `quant`, `consulting`, `policy`, etc.")

# -----------------------------
# FUNCTION: Fetch internships via SerpAPI
# -----------------------------
def fetch_internships(query, location, num_results=10):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "hl": "en",
        "api_key": serpapi_key
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        st.error(f"SerpAPI Error {res.status_code}: {res.text}")
        return []

    data = res.json()
    jobs = data.get("jobs_results", [])
    return jobs[:num_results]

# -----------------------------
# FUNCTION: Summarize job with GPT
# -----------------------------
def summarize_job(description):
    if not client:
        return "No AI summary (OpenAI key not set)."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a helpful career advisor summarizing job listings for students."},
                {"role": "user", "content": f"Summarize this internship in 3 lines, focusing on what the intern will do and learn:\n\n{description}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# MAIN APP
# -----------------------------
st.write("Fetching latest internships...")

if st.button("🚀 Search Internships"):
    jobs = fetch_internships(query, location, num_results)
    if not jobs:
        st.warning("No jobs found. Try different keywords or locations.")
    else:
        st.success(f"✅ Found {len(jobs)} internships — scroll down!")
        for job in jobs:
            title = job.get("title", "Untitled Role")
            company = job.get("company_name", "Unknown Company")
            link = job.get("apply_link") or job.get("share_link", "#")
            desc = job.get("description", "No description provided.")
            location = job.get("location", "")
            st.markdown(f"### [{title}]({link})")
            st.write(f"🏢 **{company}** — 📍 {location}")
            with st.expander("Job Description"):
                st.write(desc)
            summary = summarize_job(desc)
            st.info(summary)
            st.markdown("---")

else:
    st.info("👈 Enter your filters on the left and click **Search Internships**!")




