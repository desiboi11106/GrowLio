import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from openai import OpenAI

# ---------- CONFIG ----------
st.set_page_config(page_title="🤖 Radhe Intern Finder", layout="wide")
st.title("🎯 Radhe Intern Finder")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------- USER PROFILE ----------
st.sidebar.header("Your Profile")
profile = {
    "name": st.sidebar.text_input("Name", "Radhe Bhagat"),
    "major": st.sidebar.text_input("Major", "Economics and International Affairs"),
    "school": st.sidebar.text_input("University", "Georgia Tech"),
    "skills": st.sidebar.text_area("Skills (comma separated)", "Finance, Python, Data Analysis, Excel"),
    "interests": st.sidebar.text_area("Interests (comma separated)", "Investment Banking, Data Analytics, Consulting"),
    "location": st.sidebar.text_input("Preferred Location", "Atlanta, Remote")
}

keywords = [k.strip() for k in profile["interests"].split(",") if k.strip()]

# ---------- SCRAPER ----------
@st.cache_data
def scrape_lever_jobs(keywords):
    jobs = []
    for kw in keywords:
        url = f"https://jobs.lever.co/api/postings/?search={kw}&mode=json"
        try:
            res = requests.get(url, timeout=10)
            data = res.json()
            for d in data:
                jobs.append({
                    "company": d.get("hostedUrl", "").split(".")[0].replace("https://jobs.", ""),
                    "title": d.get("text", ""),
                    "url": d.get("hostedUrl", ""),
                    "location": d.get("categories", {}).get("location", ""),
                    "description": d.get("descriptionPlain", "")[:400]
                })
        except Exception as e:
            print(f"Error scraping {kw}: {e}")
    return pd.DataFrame(jobs)

# ---------- GPT SUMMARIZER ----------
def summarize_job(job, profile):
    prompt = f"""
    You are an AI career assistant helping a student at {profile['school']} majoring in {profile['major']}.
    Based on their interests ({profile['interests']}) and skills ({profile['skills']}), 
    summarize and rate how well this internship fits them (1-10).

    Job Title: {job['title']}
    Company: {job['company']}
    Description: {job['description']}
    Location: {job['location']}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a career assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

# ---------- MAIN ----------
st.write("Fetching latest internships...")

jobs_df = scrape_lever_jobs(keywords)

if jobs_df.empty:
    st.warning("No jobs found. Try different keywords or locations.")
else:
    for _, job in jobs_df.head(10).iterrows():
        with st.expander(f"💼 {job['title']} — {job['company']} ({job['location']})"):
            st.write(job['description'])
            summary = summarize_job(job, profile)
            st.markdown(f"**AI Summary:** {summary}")
            st.markdown(f"[Apply Here]({job['url']})")

st.success("Done ✅ — Scroll through the internships above!")


