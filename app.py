import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import re

def format_for_email(response_text):
    """
    Cleans up the AI response so it's suitable for copying into an email.
    Removes Markdown and extra spacing.
    """
    formatted = response_text.replace("**", "")  # remove bold markup
    formatted = formatted.replace("\n\n", "\n")  # remove extra spacing
    return formatted.strip()

from PIL import Image

logo = Image.open("assets/logo.png")

st.markdown(
    """
    <div style="text-align: center; padding-bottom: 10px;">
        <img src="https://raw.githubusercontent.com/RichmondChambers/richmond-immigration-assistant/main/assets/logo.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load API Key securely ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Load FAISS Index and Metadata ---
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("faiss_index.index")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_index_and_metadata()

# --- Helper: Extract Prospect Name ---
def extract_prospect_name(enquiry):
    closings = ["regards,", "best,", "sincerely,", "thanks,", "kind regards,"]
    for closing in closings:
        match = re.search(closing + r"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", enquiry, re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.search(r"my name is\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", enquiry, re.IGNORECASE)
    if match:
        return match.group(1)
    return "[Prospect]"

# --- Helper: Embed Query ---
def get_embedding(text, model="text-embedding-3-small"):
    result = openai.embeddings.create(input=[text], model=model)
    return result.data[0].embedding

# --- Helper: Search Index ---
def search_index(query, k=5):
    query_embedding = get_embedding(query)
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results

# --- Helper: Build GPT Prompt ---
def format_prompt(question, sources):
    name = extract_prospect_name(question)
    context = "\n\n---\n\n".join([src["content"] for src in sources])
    
    prompt = f"""
You are a knowledgeable and precise UK immigration legal assistant working at Richmond Chambers Immigration Barristers.

Your task is to draft a preliminary response email to a prospective client enquiry. This response must follow Richmond Chambers' formal structure, tone, and content standards. You will base your entire response strictly on the source material provided below, drawn from the internal Google Drive knowledge centre, GOV.UK, and richmondchambers.com. Do not reference or use any external or general knowledge.

You must not provide legal advice. Instead, assist with legal research and draft structured communications that:
- Identify relevant immigration categories;
- Flag eligibility or evidential issues;
- Suggest possible legal routes (with caution);
- Encourage the prospect to seek a formal consultation.

Your email must follow this format:

---

Dear {name},

Thank you for contacting Richmond Chambers Immigration Barristers.

YOUR IMMIGRATION MATTER

I understand from your enquiry that {question.strip()}

---

INITIAL THOUGHTS

Write a detailed and insightful response that demonstrates legal expertise and engages with the specific circumstances described. Identify immigration categories, legal criteria, evidential concerns, and areas where further advice is needed. Show strategic awareness, but remain cautious and grounded in the source material.

Conclude this section by encouraging the prospect to arrange an initial consultation for personalised legal advice.

---

HOW WE CAN ASSIST

Use bullet points. If scope of work examples are not found in the source material, use:

- Preparing and submitting visa applications;
- Providing strategic advice on eligibility and evidence;
- Representing clients in appeals and administrative reviews.

---

NEXT STEPS

Include the following standard closing text:

If you would like to discuss your immigration matter in more detail, I would be pleased to advise you further at an initial consultation meeting. At this meeting I will take further instructions from you, advise you in detail as to the requirements of the UK's Immigration Rules and any other relevant guidance or case law, provide you with an assessment of your prospects of success and answer any questions that you may have. Following the meeting you will receive a written summary of my advice.

A member of our administration team will be in touch by email shortly with details of all the immigration barristers we have available to advise you at an initial consultation meeting, along with information regarding our professional fees.

We look forward to hopefully having an opportunity to advise you further.

Kind regards,

---

**SOURCE MATERIAL**
{context}
"""
    return prompt

# --- Streamlit App UI ---
st.markdown(
    "<h1 style='text-align: center; font-size: 2.6rem;'>RC Immigration Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown("Enter a new enquiry below to generate an initial thoughts email.")

with st.form("query_form"):
    enquiry = st.text_area("Client Enquiry", height=250)
    submit = st.form_submit_button("Generate Response")

if submit and enquiry:
    with st.spinner("Searching documents and drafting response..."):
        results = search_index(enquiry)
        prompt = format_prompt(enquiry, results)
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        reply = completion.choices[0].message.content
    st.success("Response generated.")
    st.text_area("Draft Email", value=reply, height=600)
    # Create email-friendly version
    email_version = format_for_email(reply)
    st.subheader("📩 Email-Ready Version")
    st.text_area("Formatted for email", value=email_version, height=400)

st.text_area("Formatted for email", value=email_version, height=400)

 # Copy button
        st.markdown(
            f"""
            <button onclick="navigator.clipboard.writeText({json.dumps(email_version)})"
                    style="
                        margin-top: 10px;
                        padding: 8px 16px;
                        background-color: #0066cc;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    ">
                📋 Copy to Clipboard
            </button>
            """,
            unsafe_allow_html=True
        )


