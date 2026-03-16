import logging
import os
from dataclasses import dataclass

from .config import load_config


@dataclass
class RuntimeComponents:
    config: object
    es: object
    cross_encoder_model: object
    feature_extractor: object
    ltr_ranker: object
    query_router: object
    logger: logging.Logger


def create_es_client(config):
    from elasticsearch import Elasticsearch

    return Elasticsearch(config.elasticsearch_hosts)


def setup_structured_logger(config, logger_name="search_app"):
    from pythonjsonlogger import jsonlogger

    os.makedirs(config.log_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    target_log_file = os.path.abspath(config.log_file)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, "baseFilename", "")) == target_log_file:
                return logger

    log_handler = logging.FileHandler(config.log_file)
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    return logger


def load_cross_encoder_model(config):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.cross_encoder_model_name)


def load_ltr_components(config):
    from ranking.feature_extractor import FeatureExtractor
    from ranking.ranker import LTRRanker

    print("Initializing LTR components...")
    feature_extractor = FeatureExtractor()
    ltr_ranker = LTRRanker(feature_extractor)

    if os.path.exists(config.ltr_model_path):
        try:
            ltr_ranker.load_model(config.ltr_model_path)
            print(f"✓ LTR model loaded from {config.ltr_model_path}")
        except Exception as exc:
            print(f"✗ Failed to load LTR model: {exc}")
            ltr_ranker = None
    else:
        print(f"✗ LTR model not found at {config.ltr_model_path}")
        ltr_ranker = None

    return feature_extractor, ltr_ranker


def load_query_router(config):
    from ranking.query_router import QueryRouter

    query_router = QueryRouter(
        model_path=config.query_router_model_path,
        default_easy_mode=config.adaptive_easy_mode,
        default_hard_mode=config.adaptive_hard_mode,
    )
    query_router.easy_mode = config.adaptive_easy_mode
    query_router.hard_mode = config.adaptive_hard_mode
    query_router.hard_threshold = config.adaptive_hard_threshold

    if query_router.loaded:
        print(f"✓ Query router loaded from {config.query_router_model_path}")
    else:
        print("✗ Query router model unavailable, fallback to heuristic routing")
    print(
        "✓ Router runtime mode override: "
        f"easy={query_router.easy_mode}, hard={query_router.hard_mode}, "
        f"hard_threshold={query_router.hard_threshold:.3f}"
    )
    return query_router


def build_runtime(config=None):
    config = config or load_config()
    es = create_es_client(config)
    cross_encoder_model = load_cross_encoder_model(config)
    feature_extractor, ltr_ranker = load_ltr_components(config)
    query_router = load_query_router(config)
    logger = setup_structured_logger(config)
    return RuntimeComponents(
        config=config,
        es=es,
        cross_encoder_model=cross_encoder_model,
        feature_extractor=feature_extractor,
        ltr_ranker=ltr_ranker,
        query_router=query_router,
        logger=logger,
    )
