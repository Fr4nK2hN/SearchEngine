from sentence_transformers import SentenceTransformer

def download_models():
    """
    Downloads all necessary SentenceTransformer models.
    """
    print("Pre-downloading sentence-transformer models...")
    # Pre-download the cross-encoder model
    SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Pre-download the bi-encoder model
    SentenceTransformer("all-MiniLM-L6-v2")
    print("Sentence-transformer models downloaded successfully.")

if __name__ == "__main__":
    download_models()