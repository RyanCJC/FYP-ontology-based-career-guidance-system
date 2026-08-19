import re
import torch
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aligner = SentenceTransformer('all-mpnet-base-v2').to(device)

saved_data = torch.load('models_&_files/ontology_tensor.pt', map_location=device)
master_vocab = saved_data['vocab']
ontology_embs = saved_data['embeddings'].to(device)

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {e}"


def preserve_tech_syntax(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ' '.join(re.sub(r'[^a-z0-9\+\#\.]', ' ', text.lower()).split())


def extract_ngrams(text: str, ngram_range=(1, 3)) -> list:
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words='english',
        token_pattern=r'(?u)\.?[a-z0-9][a-z0-9\+\#\.]*'
    )
    try:
        vectorizer.fit([text])
        return vectorizer.get_feature_names_out().tolist()
    except ValueError:
        return []


OPTIMAL_THRESHOLD = 0.72

def align_resume_to_ontology(raw_text: str) -> list:
    cleaned_text = preserve_tech_syntax(raw_text)
    chunks = extract_ngrams(cleaned_text)

    if not chunks:
        return []

    chunk_embs = aligner.encode(chunks, convert_to_tensor=True, device=device)
    cosine_scores = util.cos_sim(chunk_embs, ontology_embs)
    max_scores, max_indices = torch.max(cosine_scores, dim=1)

    aligned_terms = []
    for i, score in enumerate(max_scores):
        if score.item() >= OPTIMAL_THRESHOLD:
            aligned_terms.append(master_vocab[max_indices[i].item()])

    return list(set(aligned_terms))
