import logging
import os
import fitz
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
import cohere
from concurrent.futures import ThreadPoolExecutor


def is_exist_in_vector_store_index(index, pdf_file):
    for node in index.docstore.docs.values():
        if node.metadata["file_path"] == pdf_file:
            return True
    return False


def update_vector_store_index(index, pdf_file):
    if is_exist_in_vector_store_index(index, pdf_file):
        logging.info(f"Document {pdf_file} already present in the index. Skipping.")
        return

    documents = SimpleDirectoryReader(input_files=[pdf_file]).load_data()

    for doc in documents:
        index.insert(doc)

    logging.info(f"Document {pdf_file} added to the index.")


def setup_chat_engine(directory):
    logging.info(f"Loading documents from {directory} directory")
    documents = SimpleDirectoryReader(directory).load_data()
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    embed_model = CohereEmbedding(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-english-v3.0",  # Supports all Cohere embed models
        input_type="search_query",  # Required for v3 models
    )

    logging.info("Loading LLM model")
    llm_model = Cohere(
        model="command-r",
        api_key=os.getenv("COHERE_API_KEY"),
        temperature=0.1,
        max_tokens=4000,
    )

    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.llm = llm_model

    logging.info("Building vector store index")
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, transformations=[text_splitter]
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)

    logging.info("Creating chat engine")
    chat_engine = index.as_chat_engine(
        # chat_mode="condense_plus_context",
        chat_mode="context",
        memory=memory,
        llm=llm_model,
        context_prompt=(
            "You are a chatbot, able to have normal interactions, as well as explaining research papers."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        ),
        verbose=True,
    )

    return chat_engine, index


def _get_full_text(file_path):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    full_doc_text = "\n".join([doc.text for doc in documents])
    return full_doc_text


def extract_with_llm(original_question, file_path, max_tokens=1000):
    full_doc_text = _get_full_text(file_path)

    prompt = f"""
Given the following question about a paper from the user, please provide a brief compilation of texts from the referring paper that can help answer the question.

Question:
{original_question}

Referring Paper Content:
{full_doc_text}
"""

    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0.1,
        max_tokens=max_tokens,
    )

    answer = response.text
    return answer


def improve_prompt_with_citing_context(
    original_question, citing_papers, main_paper=None, max_tokens=1000
):
    if main_paper:
        file_path = main_paper
        logging.info(f"Adding context of the main paper to the prompt.")
        full_text = extract_with_llm(
            original_question, file_path, max_tokens=max_tokens
        )
        main_paper_context = f"Current Paper Context:\n{full_text}\n"

    citation_context = "Citation Context:\n"

    with ThreadPoolExecutor() as executor:
        futures = []
        for title, file_path in citing_papers:
            logging.info(f"Adding context of {title} to the prompt.")
            future = executor.submit(
                extract_with_llm, original_question, file_path, max_tokens=max_tokens
            )
            futures.append((title, future))

        for title, future in futures:
            full_text = future.result()
            citation_context += f"Paper: {title}\n{full_text}\n"

    prompt = f"{citation_context}\nQuestion: {original_question}"

    if main_paper:
        prompt = f"{main_paper_context}\n{prompt}"

    return prompt


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from src.refextract import extract_references_from_doc_extract
    from src.rag import ensure_pdfs_are_downloaded, chat_engine

    logging.basicConfig(level=logging.INFO)

    datadir = "src/refextract/pdf_metadata/"
    file = datadir + "2307.06435-2.pdf"
    file = "/home/surya/NEU/CS5100 FAI/Project/ResearchLens/uploads/2311.17902.pdf"
    doc = fitz.open(file)
    extract = """
Direct zero-shot evaluation. For direct zero-shot evaluation,
we train DECOLA with Swin-T [39] and use Object365 data
for Phase 1, and ImageNet-21K for Phase 2 (full dataset and
classes). We compare to MDETR [26], GLIP [34], GroundingDINO [38], and MQ-Det [65] finetuned from GLIP and
GroundingDINO. Table 4 shows the results. DECOLA outperforms the previous state-of-the-arts, by 12.0/17.1 APrare
and 3.0/9.4 mAP on LVIS minival and LVIS v1.0 val, respectively. It is noteworthy that all other methods use much
richer detection labels from GoldG data [26], a collection
of grounding data (box and text expression pairs) curated
by MDETR. Furthermore, other benchmark methods show
highly imbalanced APrare and APf
in both LVIS minival and
LVIS v1.0 val (10-20 points gap). We hypothesize that the
large collection of training data coincides with LVIS vocabulary, as all data follows a natural distribution of common objects.

Highlight some of the key differences between the current paper and the related models MDETR, GLIP, GroudingDINO.
    """
    metadata = extract_references_from_doc_extract(
        doc,
        extract,
        anystyle_url="https://anystyle-webapp.azurewebsites.net/parse",
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    )

    directory = (
        "C:/Users/bayan/Desktop/Github/ResearchLens/src/refextract/pdf_metadata/"
    )
    ensure_pdfs_are_downloaded(metadata, directory)
    chat_engine(directory)
