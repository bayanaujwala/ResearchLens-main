from pathlib import Path
from src.refextract.utils import download_pdf
import logging


def ensure_pdfs_are_downloaded(metadata, directory):
    directory = Path(directory)
    available_pdfs = []

    for meta in metadata:
        if meta and "pdf_url" in meta and meta["pdf_url"]:
            safe_title = "".join(
                c for c in meta["title"] if c.isalnum() or c in " _-"
            ).rstrip()
            pdf_filename = directory / f"{safe_title}.pdf"
            if not pdf_filename.exists():
                logging.info(f"Downloading {pdf_filename}...")
                download_pdf(meta["pdf_url"], pdf_filename)
            else:
                logging.info(f"{pdf_filename} already exists.")
            available_pdfs.append(pdf_filename)

    return available_pdfs
