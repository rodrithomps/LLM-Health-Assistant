from fastapi import APIRouter
import os
import logging
import requests
from xml.etree import ElementTree as ET
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

router = APIRouter(prefix="/api/query_pubmed", tags=["PubMed Query"])

PUBMED_API_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_MAX_RESULTS = 3
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class PubMedRequest(BaseModel):
    query: str


def extract_pubmed_keywords(user_query: str) -> str:
    """
    Convierte una pregunta natural (español o inglés) en 2 a 4 términos clave
    científicos en inglés optimizados para la API de PubMed.
    """
    if not GEMINI_API_KEY:
        return user_query

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=(
                "Extract 2 to 4 concise English biomedical/sports-science search keywords "
                "for a PubMed query from the following user question. "
                "Return ONLY the keywords separated by spaces, with no quotes, punctuation, or explanation.\n\n"
                f"Question: {user_query}"
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=30,
            ),
        )
        keywords = (response.text or "").strip().replace('"', "").replace("\n", " ")
        if keywords:
            logging.info(f"PubMed translated keywords: {keywords}")
            return keywords
    except Exception as e:
        logging.error(f"Error extracting PubMed keywords: {e}")

    return user_query


def fetch_pubmed_details(pubmed_ids):
    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "xml"
    }

    response = requests.get(f"{PUBMED_API_BASE_URL}efetch.fcgi", params=params, timeout=10)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            # PMID
            pmid_elem = article.find(".//MedlineCitation/PMID")
            pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else "N/A"

            # Título completo (incluso si contiene etiquetas HTML internas como <i>)
            title_elem = article.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else "No title"

            # Año de publicación
            year_elem = article.find(".//JournalIssue/PubDate/Year")
            if year_elem is not None and year_elem.text:
                year = year_elem.text.strip()
            else:
                medline_date = article.find(".//JournalIssue/PubDate/MedlineDate")
                year = medline_date.text.strip()[:4] if medline_date is not None and medline_date.text else "Year N/A"

            # Abstract completo (une todas las secciones: Background, Methods, Results, etc.)
            abstract_elems = article.findall(".//AbstractText")
            abstract_parts = ["".join(elem.itertext()).strip() for elem in abstract_elems if "".join(elem.itertext()).strip()]
            abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"

            articles.append(f"- **{title}** ({year}) [PMID: {pmid}]\nAbstract: {abstract}")

        return {"results": articles}
    else:
        return {"results": [f"Error fetching abstracts: {response.status_code}"]}


@router.post("/")
def query_pubmed(request: PubMedRequest):
    search_term = extract_pubmed_keywords(request.query)

    params = {
        "db": "pubmed",
        "term": search_term,
        "retmode": "json",
        "retmax": PUBMED_MAX_RESULTS,
        "sort": "relevance"
    }

    response = requests.get(f"{PUBMED_API_BASE_URL}esearch.fcgi", params=params, timeout=10)

    if response.status_code == 200:
        result = response.json()
        pubmed_ids = result.get("esearchresult", {}).get("idlist", [])
        if pubmed_ids:
            return fetch_pubmed_details(pubmed_ids)
        else:
            return {"results": ["No relevant PubMed articles found."]}
    else:
        return {"results": [f"Error querying PubMed: {response.status_code}"]}