import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import re
import json
import requests
import jwt  # from PyJWT
import streamlit.components.v1 as components
from markdown_it import MarkdownIt
from index_builder import sync_drive_and_rebuild_index_if_needed, INDEX_FILE, METADATA_FILE

def google_login():
    """
    Require the user to sign in with a Google account and restrict access
    to @richmondchambers.com email addresses.
    """

    # 1. If we already have a logged-in user in this session, allow access
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    # 2. Check if Google has redirected back with a ?code=... parameter
    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0]

        # Exchange the code for tokens
        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "redirect_uri": st.secrets["GOOGLE_REDIRECT_URI"],
                "grant_type": "authorization_code",
            },
        )

        if token_response.status_code != 200:
            st.error("Authentication with Google failed. Please refresh the page and try again.")
            st.stop()

        token_data = token_response.json()
        id_token = token_data.get("id_token")

        if not id_token:
            st.error("No ID token received from Google. Access cannot be granted.")
            st.stop()

        # Decode the ID token to get the user's email address.
        # For simplicity we skip signature verification here.
        # For a stricter setup, you would verify the token using Google's public keys.
        try:
            claims = jwt.decode(id_token, options={"verify_signature": False})
        except Exception:
            st.error("Could not decode ID token. Access cannot be granted.")
            st.stop()

        email = claims.get("email", "")
        hosted_domain = claims.get("hd", "")  # sometimes set to 'richmondchambers.com'

        # Enforce @richmondchambers.com
        if email.endswith("@richmondchambers.com") or hosted_domain == "richmondchambers.com":
            st.session_state["user_email"] = email
            return email
        else:
            st.error("Access is restricted to employees of Richmond Chambers.")
            st.stop()

    # 3. If we get here, the user is not yet logged in.
    #    Show a "Sign in with Google" link that starts the OAuth flow.
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={st.secrets['GOOGLE_CLIENT_ID']}"
        f"&redirect_uri={st.secrets['GOOGLE_REDIRECT_URI']}"
        "&scope=openid%20email"
        "&prompt=select_account"
        "&access_type=offline"
    )

    st.markdown("### Richmond Chambers – Internal Tool")
    st.write("Please sign in with a Richmond Chambers Google Workspace account to access this app.")
    st.markdown(f"[Sign in with Google]({auth_url})")

    # Stop the app here until the user has logged in
    st.stop()

# --- Load API Key securely ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 🔐 Enforce Google sign-in for @richmondchambers.com
user_email = google_login()

# Optionally show who is logged in (for debugging)
# st.write(f"Signed in as: {user_email}")

# --- Load FAISS Index and Metadata ---
@st.cache_resource
def load_index_and_metadata():
    ...


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
    """
    Ensure FAISS index is up to date, then load index, metadata,
    and read last rebuilt timestamp for UI display.
    """
    sync_drive_and_rebuild_index_if_needed()

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    # Read the timestamp from drive_index_state.json
    try:
        with open("drive_index_state.json", "r") as f:
            state = json.load(f)
            last_rebuilt = state.get("last_rebuilt", "Unknown")
    except Exception:
        last_rebuilt = "Unknown"

    return index, metadata, last_rebuilt

index, metadata, last_rebuilt = load_index_and_metadata()

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
# --- Helper: Build GPT Prompts (Two-Call Architecture) ---

def build_analysis_prompt(question, sources):
    """
    First call: ask the model to prepare an internal legal analysis
    based on the enquiry and the retrieved source material.
    This is NOT shown to the client.
    """
    context = "\n\n---\n\n".join([src["content"] for src in sources])

    prompt = f"""
You are an experienced UK immigration barrister preparing an internal legal analysis
for a colleague at Richmond Chambers. This analysis is strictly for internal use only
and will not be sent to the client.

Your analysis must be grounded primarily in the source material provided from the
internal knowledge centre. You may additionally draw upon your general professional
understanding of UK immigration law to ensure coherence and accuracy. Where the
source material does not expressly address a point, identify this clearly.

If any legal or factual question cannot be assessed safely on the information
available, state that further information is required.

Maintain a consistently professional, formal tone appropriate for internal
written advice between barristers. Use precise legal terminology and avoid
colloquial phrasing. Avoid speculation or conjecture that is not supported by
the source material or by standard legal inferences.

Guidance:
- Refer to Immigration Rules, Appendices and policy only at the section or Appendix level
  (e.g. “Appendix FM”, “Appendix Skilled Worker”), not at paragraph or subparagraph level.
- Use precise, formal legal English suitable for a note between barristers.
- Do not address the client, and do not draft an email.
- Do not give a definitive view on success; your assessment is preliminary.

Please prepare a structured internal memorandum using the following headings:

1. Key Facts: (as derived from the enquiry – summarise concisely)
2. Legal Issues: (the main immigration questions arising)
3. Relevant Immigration Routes and Legal Framework:
4. Application of Law to the Facts:
5. Evidential Issues and Documentation:
6. Risks, Suitability Concerns and Discretionary Factors:
7. Further Information Required:
8. Provisional View: (preliminary only, no percentage prospects of success)

Prospect's enquiry:
\"\"\"{question.strip()}\"\"\"

SOURCE MATERIAL (internal knowledge centre – do not quote internal links or paragraph numbers):
{context}
"""
    return prompt

def build_email_prompt(question, analysis):
    """
    Second call: convert the internal legal analysis into a polished, client-facing
    'Initial Thoughts' email in the Richmond Chambers style.
    """
    name = extract_prospect_name(question)

    prompt = f"""
You are an experienced UK immigration barrister drafting a client-facing initial
response email on behalf of Richmond Chambers. Your role is to interpret the
prospect's enquiry using the internal analysis above as your primary legal basis,and drawing where appropriate on your general professional understanding of UK immigration law, to produce
a clear, natural, professional email that reflects the tone and writing style of
Richmond Chambers’ published website content and correspondence. Please now draft the full email in the required structure and tone. Do not mention that an internal analysis exists.

## Core Writing Principles (integrated requirements)
When drafting the email, you must adhere to the following professional standards:
- Maintain a consistently formal and professional tone suitable for written
  correspondence from a barrister’s chambers.
- Write in detailed, fluent prose (not rigid step-by-step analysis).
- Prioritise clarity, accuracy, and readability for a lay client, even where this
  comes at the expense of brevity.
- Use professional UK legal English, formal but expressed clearly and naturally for a lay client.
- Base your legal analysis primarily on the internal analysis and, where helpful, your general understanding of the UK immigration system, ensuring the response is natural, accurate and helpful. Where there is a conflict, follow the internal analysis.
- Identify the applicable immigration categories and sub-routes and explain the relevant legal framework in clear, client-friendly prose.
- Interpret the UK Immigration Rules and related policy where this helps to clarify the position, keeping explanations at section or Appendix level only.
- Reference legal frameworks naturally within the narrative. Cite Immigration Rules and policy only at the section or Appendix level (e.g.
  “Appendix FM”, “Appendix Skilled Worker”) and never at paragraph level.
- Identify potential eligibility or evidential issues in a client-friendly manner.
- Explain areas of legal ambiguity or discretion where relevant;
- Avoid speculative or unfounded assumptions. Do not invent new legal arguments or immigration routes. If something is not
  supported by the internal analysis, omit it or state that it cannot be assessed. Where information is incomplete,
  identify the need for clarification.
- Do not provide or imply individualised legal advice or a definitive assessment
  of prospects of success. Treat everything as preliminary commentary.
- Do not include any text from the internal analysis verbatim; paraphrase and integrate.
- Gently encourage the prospect to arrange a consultation for tailored advice. Avoid saying "strongly recommend", but "recommend" is acceptable.
- Ensure the overall email reflects the high standard of written communication
  expected from barristers at Richmond Chambers.

All section headings must be presented in **bold**.

Avoid:
- Overly rigid structure or legalistic formatting.
- Formulaic or stilted phrasing.
- Exhaustive, step-by-step legal tests.
- Cautious filler expressions such as “it appears that” or “it may be that.”
- Any expression of definitive legal advice or guaranteed outcomes.

## Required Email Structure
You must produce your output in exactly the following structure and in this exact order. 
Every heading below (including Initial Thoughts) must appear exactly as written:

---

Dear {name},

Thank you for contacting Richmond Chambers Immigration Barristers.

**Your Immigration Matter**

I understand from your enquiry that {question.strip()}

**Initial Thoughts**

This section must start with the heading **Initial Thoughts**

In the Initial Thoughts section, provide a clear, narrative explanation of the immigration routes that may be relevant to the prospect’s circumstances. Use detailed prose. Apply the relevant legal framework where helpful, but prioritise readability and clarity for the client. Do not structure this like an internal legal memorandum; instead write as a barrister writing to a client.

Provide a clear, narrative, client-friendly explanation that:
- Summarises the key facts and immigration objectives.
- Provides a clear narrative explanation of the relevant immigration routes
- Explains the applicable legal framework in a natural and helpful way.
- Applies the legal principles to the facts as described, identifying:
  - requirements that appear likely to be met,
  - points that may require clarification, and
  - potential eligibility, suitability, or evidential issues.
- Notes any strategic considerations (timing, switching routes, interaction with
  previous immigration history, suitability matters, etc.) where relevant.
- States where further information or documentation would be required before
  firm advice could be provided.
- Gently encourages the prospect to arrange an initial consultation for tailored advice.

The bullet points in these instructions are for guidance only. Do NOT output any bullet points in the Initial Thoughts section of the email.

**How We Can Assist**

At Richmond Chambers, our professional services can include:

Use 5 or 6 bullet points here.

Draw upon the information in the internal knowledge folder called 'Scopes of Work' and files called 'Scope of Work - ' for examples of professional services.  Tailor the services to the scenario.

Do not use “you” or “your” in the bullet points.

**Next Steps**

Include the following standard closing text:

If you would like to discuss your immigration matter in more detail, I would be pleased to provide further advice at an initial consultation meeting. During this meeting, I will take detailed instructions from you, explain the relevant requirements of the UK’s Immigration Rules and any applicable guidance or case law, assess the prospects of success in your case, and answer any questions you may have. After the consultation, you will receive a written summary of my advice.

A member of our administration team will contact you by email shortly with details of all the immigration barristers that we have available for an initial consultation, together with information about our professional fees.

We look forward to hopefully having an opportunity to advise you further.

Kind regards,

---

INTERNAL ANALYSIS (for your reference only – do not quote or reproduce this section in the email):

{analysis}

Using only the internal analysis above as your legal basis, please now draft the full email in the required structure and tone. Do not mention that an internal analysis exists.
"""
    return prompt

# --- Streamlit App UI ---
st.markdown(
    "<h1 style='text-align: center; font-size: 2.6rem;'>Initial Thoughts Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='color: grey; text-align: center; font-size: 0.9rem;'>Immigration law knowledge last rebuilt from Drive on: <b>{last_rebuilt}</b></p>",
    unsafe_allow_html=True
)

st.markdown("Paste a new enquiry below to generate a first draft of your initial thoughts email.")

with st.form("query_form"):
    enquiry = st.text_area("Prospect's Enquiry", height=250)
    submit = st.form_submit_button("Generate Response")

if submit and enquiry:
    with st.spinner("Searching documents and drafting response..."):
        # Step 1: retrieve relevant documents
        results = search_index(enquiry)

        # Step 2: first call – internal legal analysis
        analysis_prompt = build_analysis_prompt(enquiry, results)
        analysis_completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.2
        )
        internal_analysis = analysis_completion.choices[0].message.content

        # Step 3: second call – client-facing email based on the analysis
        email_prompt = build_email_prompt(enquiry, internal_analysis)
        email_completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": email_prompt}],
            temperature=0.3
        )
        reply = email_completion.choices[0].message.content

        st.success("Response generated.")

        # 🔹 INTERNAL ANALYSIS FIRST (on top)
        with st.expander("Internal Legal Analysis (not to be sent to prospect)", expanded=False):
            st.markdown(internal_analysis)

        # 🔹 DRAFT EMAIL SECOND (underneath)
        st.subheader("Draft Email to Prospect")
        st.text_area("Draft Email", value=reply, height=600)

        st.markdown(
    """
    ---  
    **Professional Responsibility Statement**

    AI-generated content must not be relied upon without human review. Where such
    content is used, the barrister is responsible for verifying and ensuring the accuracy
    and legal soundness of that content. AI tools are used solely to support drafting and
    research; they do not replace the barrister’s independent judgment, analysis, or duty
    of care.
    """,
    unsafe_allow_html=False,
)

        # ✅ Convert Markdown reply to HTML for the copy button
        md = MarkdownIt()
        html_reply = md.render(reply)

        components.html(
            f"""
            <style>
            .copy-button {{
                margin-top: 10px;
                padding: 8px 16px;
                background-color: #2e2e2e;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.2s ease, transform 0.1s ease;
            }}
            .copy-button:hover {{
                background-color: #4a4a4a;
            }}
            .copy-button:active {{
                background-color: #3a3a3a;
                transform: scale(0.98);
            }}
            </style>

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
