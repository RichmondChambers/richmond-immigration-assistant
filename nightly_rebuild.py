"""Scheduled entrypoint to refresh the FAISS index overnight (UK time).

Run this via cron/Cloud Scheduler at an off-peak hour (e.g. 02:00 Europe/London)
so daytime users never trigger a long rebuild inside the Streamlit app.
"""

from index_builder import sync_drive_and_rebuild_index_if_needed, is_within_rebuild_window


def main():
    rebuilt = sync_drive_and_rebuild_index_if_needed(
        respect_cooldown=False,
        respect_rebuild_window=True,
    )

    if rebuilt:
        print("Knowledge base was rebuilt from Google Drive.")
    else:
        if not is_within_rebuild_window():
            print(
                "Nightly job ran outside the permitted window; no rebuild attempted."
            )
        else:
            print("No Drive changes detected; existing index kept.")


if __name__ == "__main__":
    main()
