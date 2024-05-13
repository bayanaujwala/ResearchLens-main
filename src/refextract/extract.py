from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
import re
from rapidfuzz import fuzz
from tqdm import tqdm
import requests
import logging
from semanticscholar import SemanticScholar


def _get_extract_blocks_from_doc(extract, doc, fuzzy_threshold=0.9, min_text_len=10):
    """
    Extracts text blocks from a document that match a given extract.

    Args:
        extract (str): The text to match against the text blocks.
        doc: The fitz document object representing the document to extract from.
        fuzzy_threshold (float, optional): The minimum fuzzy match score required for a block to be considered a match. Defaults to 0.9.
        min_text_len (int, optional): The minimum length of the text block required for it to be considered a match. Defaults to 10.

    Returns:
        list: A list of tuples containing the matching text blocks, their page numbers, and their match scores.

    """
    matches = []

    for page_num in tqdm(
        range(len(doc)), leave=False, desc="Extracting matching text from document"
    ):
        page = doc.load_page(page_num)  # load the current page
        text_blocks = (
            page.get_text_blocks()
        )  # get a list of text blocks on the current page
        for block in text_blocks:
            text = block[4]

            match_score = fuzz.partial_ratio(extract, text)

            if match_score >= fuzzy_threshold and len(text) >= min_text_len:
                matches.append((block, page_num, match_score))

    matches = sorted(matches, key=lambda x: x[2], reverse=True)

    return matches


def _get_links_in_block(doc, block, page):
    # get the matched region
    bbox = fitz.Rect(block[:4])

    # get the citation links
    matched_links = []

    for link in doc[page].get_links():
        if link["kind"] == 4:  # internal links
            link_bbox = link["from"]
            if bbox.intersects(link_bbox):
                link["from_page"] = page
                matched_links.append(link)

    return matched_links


def _get_references_from_matches(doc, matches, extract):
    """
    Given text block matches of an extract in a document,
    extracts the references from the matched text blocks.

    Function first extracts links from the matched blocks,
    then uses the citation numbers (not names as of now)
    to extract the references from the bibliography
    section of the paper.

    Any citations present in matches that are not present
    in the extract are filtered out.
    """

    # Get links in the matches
    matched_links = []

    for match in matches:
        matched_links.extend(_get_links_in_block(doc, match[0], match[1]))

    # filter the links and update it with citation number
    matched_links_filtered = []

    for link in matched_links:
        # keep only citations, not equations and figures
        if not link["nameddest"].startswith("cite."):
            continue

        citation_num = doc[link["from_page"]].get_text("text", clip=link["from"])
        citation_num = re.findall(r"\d+", citation_num)

        if "page" not in link:
            logging.warning(f"Page link not found for {citation_num}")
            continue

        if len(citation_num) == 0:
            continue

        citation_num = citation_num[0]

        if citation_num not in extract:
            continue

        link["citation_number"] = citation_num
        matched_links_filtered.append(link)

    # Remove duplicate citations
    unique_matches = {}

    for match in matched_links_filtered:
        citation_num = match["citation_number"]

        # already exist, duplicate - keep if a link has more attributes than an existing one
        if citation_num in unique_matches and len(match.keys()) < len(
            unique_matches[citation_num]
        ):
            continue

        # doesn't exist
        unique_matches[citation_num] = match

    matched_links_filtered = list(unique_matches.values())

    # Get the reference corresponding to each citation
    matched_references = []

    for link in matched_links_filtered:
        linked_page = doc.load_page(link["page"])
        text_blocks = linked_page.get_text("blocks")
        citation_num = link["citation_number"]
        num_pat = r"\b" + citation_num + r"\b"

        for text in text_blocks:
            # citation number should be present in the initial section of the reference
            if re.search(num_pat, text[4][:15]):
                matched_references.append(text[4].strip())

    # Remove line breaks
    matched_references = list(map(lambda x: x.replace("\n", " "), matched_references))
    matched_references = list(map(lambda x: x.replace("- ", ""), matched_references))

    return matched_references


def _get_title_from_reftext(
    reftext,
    anystyle_url="http://localhost:4567/parse",
    request_timeout=3,
    min_title_len=15,
):
    """
    Extracts the title from the given reference text using the Anystyle API.

    Args:
        reftext (str): The reference text from which to extract the title.
        min_title_len (int, optional): The minimum length of the extracted title. Defaults to 15.
        anystyle_url (str, optional): The URL of the Anystyle API. Defaults to 'http://localhost:4567/parse'.

    Returns:
        str: The extracted title.

    Raises:
        AssertionError: If the extracted title is shorter than the minimum title length.

    """
    reftext = reftext.encode("utf-8")
    response = requests.post(
        anystyle_url,
        headers={"Content-Type": "text/plain"},
        data=reftext,
        timeout=request_timeout,
    )
    parsed_data = response.json()

    parsed_data = parsed_data[0]
    title = parsed_data.get("title")
    if title is None:
        logging.info(f"Title not found in reference text: {reftext}")
        return title

    title = " ".join(title)

    assert len(title) >= min_title_len

    return title


def _get_metadata_of_references(
    references, anystyle_url, semantic_scholar_api_key, request_timeout
):
    """
    Retrieves metadata for a list of text of reference.

    Args:
        references (list): A list of reference texts.
        anystyle_url (str): The URL of the Anystyle service.
        semantic_scholar_api_key (str): The API key for the Semantic Scholar service.
        request_timeout (int): The timeout for HTTP requests.

    Returns:
        list: A list of dictionaries containing the metadata for each reference.
    """

    # get title from reference text
    def _get_title_from_reftext_wrapper(reftext):
        return _get_title_from_reftext(reftext, anystyle_url, request_timeout)

    with ThreadPoolExecutor() as executor:
        references_title = list(
            tqdm(
                executor.map(_get_title_from_reftext_wrapper, references),
                total=len(references),
                desc="Extracting titles from references",
            )
        )

    # search for the metadata using the paper title
    sch = SemanticScholar(api_key=semantic_scholar_api_key, timeout=request_timeout)

    def _search_paper_wrapper(title):
        if title is None:
            return None
        results = sch.search_paper(
            title, limit=1, fields=["title", "paperId", "externalIds", "openAccessPdf"]
        )

        if results.total == 0:
            logging.info(f'No metadata found for paper "{ref}"')
            return None

        meta = results[0]
        return meta.raw_data

    with ThreadPoolExecutor() as executor:
        references_meta = list(
            tqdm(
                executor.map(_search_paper_wrapper, references_title),
                total=len(references_title),
                desc="Fetching metadata",
            )
        )

    # get the pdf URL of each paper
    for meta in references_meta:
        if meta is None:
            continue

        if meta["openAccessPdf"] is not None:
            meta["pdf_url"] = meta["openAccessPdf"]["url"]
        elif "ArXiv" in meta["externalIds"]:
            meta["pdf_url"] = f"https://arxiv.org/pdf/{meta['externalIds']['ArXiv']}"
        else:
            meta["pdf_url"] = None

    return references_meta


def extract_references_from_doc_extract(
    doc,
    extract,
    semantic_scholar_api_key,
    anystyle_url="http://localhost:4567/parse",
    request_timeout=10,
    fuzzy_threshold=92,
    min_matched_text_len=10,
):
    if isinstance(doc, str):
        doc = fitz.open(doc)
    elif isinstance(doc, fitz.Document):
        pass
    else:
        raise ValueError("doc should be a file path or a fitz Document object")

    matched_blocks = _get_extract_blocks_from_doc(
        extract,
        doc,
        fuzzy_threshold,
        min_matched_text_len,
    )
    logging.info(f"Found {len(matched_blocks)} matching text blocks")
    references = _get_references_from_matches(doc, matched_blocks, extract)

    # log extracted citations
    for ref in references:
        logging.info(f"Extracted reference: {ref}")

    metadata = _get_metadata_of_references(
        references, anystyle_url, semantic_scholar_api_key, request_timeout
    )

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datadir = "/home/surya/NEU/CS5100 FAI/Project/pdfreader/"
    file = datadir + "test2.pdf"
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
        fuzzy_threshold=80,
    )

    print("Extracted titles:")
    for meta in metadata:
        print(meta["title"])

    print("Extracted pdf URLs:")
    for meta in metadata:
        print(meta["pdf_url"])
