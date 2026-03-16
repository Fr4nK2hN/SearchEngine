from sentence_transformers import CrossEncoder, SentenceTransformer

from ..config import load_config


def download_models():
    """Download the configured sentence-transformer models."""
    config = load_config()
    print("Pre-downloading sentence-transformer models...")
    CrossEncoder(config.cross_encoder_model_name)
    SentenceTransformer(config.bi_encoder_model_name)
    print("Sentence-transformer models downloaded successfully.")
