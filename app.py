import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import re
import json
import streamlit.components.v1 as components
from markdown import markdown

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
    """
    Builds a detailed prompt for drafting a prospect‑response email
    by Richmond Chambers Immigration Barristers. Uses only supplied
    source materials from the internal knowledge centre.
    """
    name = extract_prospect_name(question)
    context = "\n\n---\n\n".join([src["content"] for src in sources])
    
    prompt = f"""
You are a knowledgeable and precise UK immigration legal assistant. You read and analyse enquiries from new prospects and assesses them against the UK's immigration rules, Home Office caseworker guidance, policy documents, and UK immigration case law. Your primary function is to support a qualified immigration barrister by helping interpret the Immigration Rules and draft well-informed, clearly structured initial response emails to enquiries from potential clients. Your response must follow Richmond Chambers' formal structure, tone, and content standards. You will base your entire response strictly on the source material provided below, drawn from the internal Google Drive knowledge centre. Do not reference or use any external or general knowledge.

You must not provide legal advice, but you will: 
- Interpret the UK Immigration Rules and related policy; 
- Identify applicable immigration categories; 
- Flag eligibility and evidential issues; 
- Suggest possible pathways (with caution) based on the individual's circumstances described in the enquiry;  
- Highlight where further advice is required;
- Gently encourage the prospect to seek a formal consultation.

Maintain a consistently professional and formal tone, suitable for written correspondence from a barrister’s chambers.

Prioritise clarity, caution, and accuracy at all times, ensuring that responses are precise and unambiguous, even where this comes at the expense of brevity.

Use formal legal English throughout all outputs.

Avoid speculation or assumptions and refrain from expressing opinions or conjecture.

Employ cautious, neutral, and accurate language, clearly identifying where further information or professional legal advice would be required before reaching a conclusion.

Use precise legal terminology and avoid colloquial phrasing.

Present all section headings in bold.

Ground all responses in the organisation’s internal knowledge centre.

Cite legislation, Appendices, and case law only at the section or Appendix level (e.g. “Appendix FM”), avoiding citation at the paragraph or subparagraph level (e.g. “paragraph 12(a)”).

Never provide or imply individualised legal advice under any circumstances.

The overriding objective is to produce measured, accurate, and professionally appropriate responses that reflect the standards of written communication expected from barristers at Richmond Chambers.

**Opening**
Your email must follow this format:

---

Dear {name},

Thank you for contacting Richmond Chambers Immigration Barristers.

YOUR IMMIGRATION MATTER

I understand from your enquiry that {question.strip()}

Conclude this section by inserting this heading:

INITIAL THOUGHTS

Write this section in well‑structured prose (no bullet points here).  

Write a detailed and insightful response that demonstrates legal expertise and engages with the specific circumstances described. 

Reference relevant Immigration Rules, Appendices, or policy documents at the section level.

For this section:  
- Identify the applicable immigration route or routes (e.g., Appendix FM, Skilled Worker route, Global Talent route) and the legal criteria.  
- Highlight at least two key eligibility or evidential issues relevant to the prospect’s specific scenario.  
- Identify any areas where further information is needed from the prospect.
- Provide strategic commentary (for example, timing issues, switch of route, risk factors) and mention the firm’s expertise (e.g., “At Richmond Chambers our immigration team”).  
- Use cautious, professional legal English.  
- Remain grounded in the legal material.
- Encourage the reader to arrange an initial consultation for personalised legal advice.

Conclude this section by gently encouraging the prospect to arrange an initial consultation for personalised legal advice. You must not use the word 'strongly' in the concluding sentence.

HOW WE CAN ASSIST

At Richmond Chambers, we offer a range of professional services, including:

Use bullet points here.

Draw upon uploaded scope of work documents for examples of services.

Do not use “you” or “your” in the bullet points.

Based on your knowledge centre or, if not available, use the following as fallback:  
- Preparing and submitting visa applications;  
- Providing strategic advice on eligibility and evidence;  
- Representing clients in appeals and administrative reviews;  
- (Add any additional service tailored to scenario).

NEXT STEPS

Include the following standard closing text:

If you would like to discuss your immigration matter in more detail, I would be pleased to provide further advice at an initial consultation meeting. During this meeting, I will take detailed instructions from you, explain the relevant requirements of the UK’s Immigration Rules and any applicable guidance or case law, assess the prospects of success in your case, and answer any questions you may have. After the consultation, you will receive a written summary of my advice.

A member of our administration team will contact you by email shortly with details of all the immigration barristers that we have available for an initial consultation, together with information about our professional fees.

We look forward to hopefully having an opportunity to assist you further.

Kind regards,

---

**SOURCE MATERIAL (for reference only – do not cite internal page links or paragraph numbers):**
{context}

Please begin drafting the full email now, following the structure and tone described.
"""
    return prompt

# --- Streamlit App UI ---
st.markdown(
    "<h1 style='text-align: center; font-size: 2.6rem;'>RC Immigration Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown("Enter a new enquiry below to generate a first draft of your initial thoughts email.")

with st.form("query_form"):
    enquiry = st.text_area("Prospect's Enquiry", height=250)
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

# ✅ Only render this after 'reply' is created
html_reply = markdown(reply)  # Convert Markdown to HTML

components.html(
    f"""
    <button class="copy-button" onclick="copyToClipboard()">📋 Copy to Clipboard</button>

    <script>
    async function copyToClipboard() {{
        const htmlContent = `{html_reply.replace("`", "\\`")}`;
        const plainText = `{reply.replace("`", "\\`")}`;

        const blobHtml = new Blob([htmlContent], {{ type: 'text/html' }});
        const blobText = new Blob([plainText], {{ type: 'text/plain' }});

        const clipboardItem = new ClipboardItem({{
            'text/html': blobHtml,
            'text/plain': blobText
        }});

        await navigator.clipboard.write([clipboardItem]);
        alert("Formatted text copied! Paste into Gmail or Google Docs to retain formatting.");
    }}
    </script>
    """,
    height=120,
    scrolling=False
)

