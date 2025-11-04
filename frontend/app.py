import streamlit as st
import pandas as pd
import requests
from io import StringIO

# ---------- Config ----------
API_BASE = "http://127.0.0.1:5000"
st.set_page_config(page_title="🎫 Ticket Uploader & Analyzer", layout="wide")

# ---------- UI ----------
st.title("🎫 Ticket Upload & Analysis")

# --- Sidebar filters/search ---
st.sidebar.header("Filters & Search")
search_query = st.sidebar.text_input("Search tickets (keyword)")
priority_filter = st.sidebar.selectbox("Priority filter", ["All", "High", "Medium", "Low"])
search_button = st.sidebar.button("🔎 Search Tickets")

# --- File uploader ---
st.subheader("1️⃣ Upload ticket (then press Upload button)")
uploaded_file = st.file_uploader("Choose a .txt or .csv file", type=["txt", "csv"])

# --- Preview ---
preview_content = ""
if uploaded_file:
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".txt"):
        preview_content = uploaded_file.read().decode("utf-8")
        st.text_area("Ticket preview", preview_content, height=200)
    else:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(StringIO(uploaded_file.read().decode("utf-8")))
            st.dataframe(df)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# --- Upload button ---
if st.button("📤 Upload Ticket"):
    if not uploaded_file:
        st.warning("Please choose a file first.")
    else:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_file.seek(0)
                df = pd.read_csv(StringIO(uploaded_file.read().decode("utf-8")))
                if "content" not in df.columns:
                    st.error("CSV must contain a 'content' column to upload rows.")
                else:
                    successes, failures = 0, 0
                    for _, row in df.iterrows():
                        content = str(row["content"]).strip()
                        if not content:
                            failures += 1
                            continue
                        payload = {"filename": uploaded_file.name, "content": content}
                        resp = requests.post(f"{API_BASE}/upload", json=payload, timeout=30)
                        if resp.ok:
                            successes += 1
                        else:
                            failures += 1
                    st.success(f"✅ Uploaded {successes} rows. ❌ Failed: {failures}.")
            else:
                uploaded_file.seek(0)
                content = uploaded_file.read().decode("utf-8").strip()
                if not content:
                    st.warning("File is empty.")
                else:
                    payload = {"filename": uploaded_file.name, "content": content}
                    resp = requests.post(f"{API_BASE}/upload", json=payload, timeout=30)
                    if resp.ok:
                        st.success(f"✅ Ticket '{uploaded_file.name}' saved successfully!")
                    else:
                        st.error(f"Upload failed: {resp.text}")
        except Exception as e:
            st.error(f"Error uploading ticket(s): {e}")

# --- Analyze ticket text manually ---
st.subheader("2️⃣ Analyze a ticket text")
manual_text = st.text_area(
    "Paste ticket text here (or leave empty to use uploaded preview):",
    value=preview_content,
    height=200
)
if st.button("🔍 Analyze Text (Predict Priority)"):
    if not manual_text.strip():
        st.warning("Add some text to analyze.")
    else:
        try:
            resp = requests.post(f"{API_BASE}/analyze", json={"content": manual_text.strip()}, timeout=30)
            if resp.ok:
                d = resp.json()
                st.success(f"Predicted Priority: {d.get('priority', 'Unknown')}")
                st.json(d)
            else:
                st.error(f"Analysis failed: {resp.text}")
        except Exception as e:
            st.error(f"Error analyzing ticket: {e}")

st.markdown("---")

# --- Browse stored tickets ---
if search_button:
    try:
        params = {"search": search_query.strip(), "priority": priority_filter}
        resp = requests.get(f"{API_BASE}/tickets", params=params, timeout=30)
        if resp.ok:
            tickets = resp.json()
            if tickets:
                df = pd.DataFrame(tickets)
                st.dataframe(df, use_container_width=True)
                st.write(f"Total: {len(df)} tickets")
                if "priority" in df.columns:
                    st.bar_chart(df["priority"].value_counts())
            else:
                st.info("No tickets matched your criteria.")
        else:
            st.error(f"Failed to fetch tickets: {resp.text}")
    except Exception as e:
        st.error(f"Error fetching tickets: {e}")

st.markdown("---")
st.title("🎯 Ticket Solution Recommendation System")

# --- Text input ---
ticket_text = st.text_area(
    "Enter your ticket description:",
    height=200,
    placeholder="Describe your issue or problem here..."
)

# --- Recommend Solution button ---
if st.button("🔍 Recommend Solution"):
    if not ticket_text.strip():
        st.warning("⚠️ Please enter a ticket description first.")
    else:
        try:
            with st.spinner("Finding best-matching solution..."):
                response = requests.post(
                    f"{API_BASE}/recommend",
                    json={"ticket_text": ticket_text}
                )

            if response.status_code == 200:
                data = response.json()
                
                if 'recommended_solution' in data:
                    st.success("✅ Recommended Solution:")
                    st.write(data['recommended_solution'])
                    st.caption(f"🧠 Similarity Score: {data.get('similarity_score', 'N/A')}")
                else:
                    st.error("No recommendation found in response.")
                    st.json(data)

            else:
                st.error(f"Backend returned {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Error contacting backend: {e}")