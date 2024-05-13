import os
import logging
import cohere
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.serving import is_running_from_reloader
from huggingface_hub import InferenceClient

import sys

from werkzeug.utils import secure_filename

sys.path.append(".")

from src.rag import (
    setup_chat_engine,
    ensure_pdfs_are_downloaded,
    update_vector_store_index,
    improve_prompt_with_citing_context,
)
from src.refextract import extract_references_from_doc_extract


app = Flask(__name__)
app.logger.setLevel(level=logging.INFO)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploads"
UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Setup the RAG chat engine
# chat_engine, vectordb = setup_chat_engine(UPLOAD_FOLDER)
from werkzeug.serving import is_running_from_reloader

# Setup the RAG chat engine
chat_engine, vectordb = None, None
if not is_running_from_reloader():
    chat_engine, vectordb = setup_chat_engine(UPLOAD_FOLDER)


@app.route("/")
def index():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return render_template("chat.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    # Check if the post request has the file part
    if "pdf_file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["pdf_file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and file.content_type == "application/pdf":
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        update_vector_store_index(vectordb, filepath)
        return redirect(url_for("index"))
    else:
        flash("Invalid file format")
        return redirect(request.url)


def _rag_response(input):
    # get response from RAG
    response = chat_engine.chat(input)

    # app.logger.debug(f"Retrieved {len(response.source_nodes)} Source nodes")
    # for src in response.source_nodes:
    #     app.logger.debug(src)

    response = response.response
    return response


def _cohere_response(input):
    # get response from simple cohere
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = co.chat(
        message=input,
        model="command-r",
        temperature=0.1,
        max_tokens=4000,
    )
    response = response.text
    return response


def _llama_response(input):
    client = InferenceClient(
        model="https://u6al5xke1es2ir4v.us-east-1.aws.endpoints.huggingface.cloud",
        token=os.getenv("HUGGINGFACE_API_KEY"),
    )
    response = client.text_generation(input, max_new_tokens=500, temperature=0.05)
    return response


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    filename = request.form.get("filename")
    input = msg

    method = request.form.get("contentType", "text")
    app.logger.info(f"Using {method} model")
    if method == "text":
        _chat_response = _cohere_response
    elif method == "math":
        _chat_response = _llama_response
        response = _chat_response(input)
        return response

    app.logger.info(f"Received message: {msg} for {filename}")

    if filename is None or filename == "":
        response = _chat_response(input)
        return response

    # check if text contains references
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    metadata = extract_references_from_doc_extract(
        filepath,
        input,
        anystyle_url="https://anystyle-webapp.azurewebsites.net/parse",
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        request_timeout=20,
        fuzzy_threshold=80,
    )
    metadata = [m for m in metadata if m is not None]
    pdfs = ensure_pdfs_are_downloaded(metadata, UPLOAD_FOLDER)
    app.logger.info(f"Using PDFs: {' '.join(map(str, pdfs))}")

    referenced_papers = [m["title"] for m in metadata if m is not None]

    # Needed for RAG
    # update vector store if it does not exist
    # for pdf in pdfs:
    #     update_vector_store_index(vectordb, pdf)

    # citing papers
    citing_papers = [(title, pdf) for title, pdf in zip(referenced_papers, pdfs)]

    # Use our GAG approach
    improved_prompt = improve_prompt_with_citing_context(
        input, citing_papers, main_paper=filepath
    )
    app.logger.debug(f"Improved prompt: {improved_prompt}")

    # get response from chat LLM
    response = _chat_response(improved_prompt)

    # Add references to the response
    referenced_papers = "\n".join(set(referenced_papers))

    reference_template = f"Referenced papers:\n{referenced_papers}"
    response = response + "\n\n" + reference_template

    return response
