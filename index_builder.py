import os
import json
from datetime import datetime

# You will also need your Google Drive client imports here, e.g.
# from google.oauth2 import service_account
# from googleapiclient.discovery import build

DRIVE_FOLDER_ID = "13J-DiERhtS1VWgF2GtZ1wnMfbUzkq6-G"
STATE_FILE = "drive_index_state.json"  # tracks last seen file states
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"


def get_drive_service():
    """
    Build an authenticated Google Drive API client.
    You should adapt this to however you currently authenticate
    (service account via st.secrets, etc.).
    """
    # Example with service account JSON in st.secrets:
    #
    # credentials = service_account.Credentials.from_service_account_info(
    #     st.secrets["gcp_service_account"],
    #     scopes=["https://www.googleapis.com/auth/drive.readonly"],
    # )
    # return build("drive", "v3", credentials=credentials)
    #
    # For now, leave as a placeholder:
    raise NotImplementedError("Implement get_drive_service() using your existing Drive auth code.")


def list_drive_files():
    """
    Return a list of files (id, name, modifiedTime) in the target folder.
    """
    service = get_drive_service()
    query = f"'{13J-DiERhtS1VWgF2GtZ1wnMfbUzkq6-G}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields="files(id, name, modifiedTime, mimeType)"
    ).execute()
    return results.get("files", [])


def load_previous_state():
    """
    Load last seen file state from JSON (if it exists).
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    """
    Save current file state to JSON.
    """
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def have_files_changed(current_files, previous_state):
    """
    Compare current Drive files with previous state to see if anything
    is new or modified.
    """
    current_state = {f["id"]: f["modifiedTime"] for f in current_files}
    if current_state != previous_state.get("files", {}):
        return True, current_state
    return False, current_state


def rebuild_index_from_drive(files):
    """
    Download files, extract text, chunk, embed, and rebuild FAISS + metadata.

    IMPORTANT: This is where you plug in your existing index-building code
    from Colab. The structure should be:

    - Download each file
    - Extract text
    - Split into chunks
    - Get embeddings (OpenAI)
    - Build FAISS index
    - Save FAISS index to INDEX_FILE
    - Save metadata (list of dicts) to METADATA_FILE
    """
    # TODO: copy your existing "build index" logic here.
    # make sure each metadata entry includes at least: { "content": "...", "source_file_id": "...", ... }
    raise NotImplementedError("Implement rebuild_index_from_drive() with your existing index-building code.")


def sync_drive_and_rebuild_index_if_needed():
    """
    Check Drive for new/updated files; if changes are detected,
    rebuild FAISS index and metadata from scratch.
    """
    # 1. List current files in the Drive folder
    files = list_drive_files()

    # 2. Load last recorded state
    previous_state = load_previous_state()

    # 3. Compare
    changed, current_state = have_files_changed(files, previous_state)

    if changed or not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        # Something changed or index doesn't exist: rebuild
        # You might want to show a message in Streamlit; we can do that in app.py
        rebuild_index_from_drive(files)

        # Save new state
        save_state({"files": current_state})
        return True  # indicates index was rebuilt

    return False  # no changes
