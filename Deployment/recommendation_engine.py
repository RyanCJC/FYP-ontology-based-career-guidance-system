import faiss
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    ft_model = SentenceTransformer('models_&_files/mpnet-base-onet-recsys-finetuned').to(device)
    faiss_index = faiss.read_index('models_&_files/onet_faiss.index')
    onet_metadata = pd.read_pickle('models_&_files/onet_metadata.pkl')
except Exception as e:
    print(f"Model loading error: {e}. Ensure all files are in the /models_&_files directory.")


def get_top_matches(aligned_skills: list, top_k: int = 10):
    if not aligned_skills:
        return []

    query_text = " ".join(aligned_skills)
    q_emb = ft_model.encode([query_text], normalize_embeddings=True, convert_to_tensor=False)
    distances, indices = faiss_index.search(np.array(q_emb).astype('float32'), top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        score = distances[0][i]
        row = onet_metadata.iloc[idx]
        results.append({
            "Rank": i + 1,
            "Occupation Title": row['Title'],
            "O*NET Code": row['ONET_SOC_Code'],
            "Match Confidence": score
        })

    return results

def get_single_job_score(aligned_skills: list, target_onet_code: str) -> float:
    """Score one specific job directly without searching the full index."""
    if not aligned_skills:
        return 0.0

    # Find the row position of the target job in metadata
    matches = onet_metadata.index[onet_metadata['ONET_SOC_Code'] == target_onet_code].tolist()
    if not matches:
        return 0.0

    idx = matches[0]

    # Encode the simulated skills
    query_text = " ".join(aligned_skills)
    q_emb = ft_model.encode([query_text], normalize_embeddings=True, convert_to_tensor=False)
    q_vec = np.array(q_emb).astype('float32')

    # Reconstruct the job vector from FAISS and compute dot product directly
    job_vec = np.zeros((1, faiss_index.d), dtype='float32')
    faiss_index.reconstruct(idx, job_vec[0])

    score = float(np.dot(q_vec[0], job_vec[0]))
    return score