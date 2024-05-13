import requests
from pathlib import Path


def download_pdf(url, filename):
    response = requests.get(url)
    file = Path(filename)
    file.write_bytes(response.content)
