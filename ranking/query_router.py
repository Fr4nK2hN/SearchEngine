import os
import pickle
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}
QUESTION_WORDS = {"what", "when", "where", "which", "who", "why", "how"}

QUERY_FEATURE_NAMES = [
    "query_len_chars",
    "query_terms",
    "unique_terms",
    "stopword_ratio",
    "prefix_like",
    "unique_ratio",
]
RETRIEVAL_FEATURE_NAMES = [
    "base_top_score",
    "base_second_score",
    "base_score_gap",
    "base_score_ratio",
    "base_top10_mean",
    "base_top10_std",
    "result_count",
]
FEATURE_NAMES = QUERY_FEATURE_NAMES
EXTENDED_FEATURE_NAMES = QUERY_FEATURE_NAMES + RETRIEVAL_FEATURE_NAMES

EXPERT_TO_MODE = {
    "Baseline": "baseline",
    "LTR": "ltr",
    "Cross-Encoder": "cross_encoder",
    "Hybrid": "hybrid",
}


def tokenize_query(query):
    return TOKEN_PATTERN.findall((query or "").lower())


def stopword_ratio(tokens):
    if not tokens:
        return 1.0
    return sum(1 for t in tokens if t in STOPWORDS) / len(tokens)


def is_prefix_like_query(tokens):
    if not tokens:
        return True
    if len(tokens) <= 2:
        return True
    if len(tokens) <= 3 and tokens[-1] in STOPWORDS:
        return True
    return False


def query_feature_vector(query):
    tokens = tokenize_query(query)
    uniq = set(tokens)
    return [
        float(len(query or "")),
        float(len(tokens)),
        float(len(uniq)),
        float(stopword_ratio(tokens)),
        1.0 if is_prefix_like_query(tokens) else 0.0,
        float((len(uniq) / len(tokens)) if tokens else 0.0),
    ]


def retrieval_feature_vector(results):
    if not results:
        return [0.0 for _ in RETRIEVAL_FEATURE_NAMES]

    scores = []
    for hit in results[:10]:
        try:
            scores.append(float(hit.get("_score", 0.0) or 0.0))
        except (AttributeError, TypeError, ValueError):
            scores.append(0.0)

    if not scores:
        return [0.0 for _ in RETRIEVAL_FEATURE_NAMES]

    top_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0.0
    score_gap = top_score - second_score
    score_ratio = (second_score / top_score) if top_score > 0 else 0.0
    score_mean = sum(scores) / len(scores)
    score_var = sum((score - score_mean) ** 2 for score in scores) / len(scores)
    return [
        float(top_score),
        float(second_score),
        float(score_gap),
        float(score_ratio),
        float(score_mean),
        float(score_var ** 0.5),
        float(len(results)),
    ]


def router_feature_vector(query, results=None, feature_names=None):
    feature_names = list(feature_names or FEATURE_NAMES)
    feature_map = {}
    feature_map.update(zip(QUERY_FEATURE_NAMES, query_feature_vector(query)))
    feature_map.update(zip(RETRIEVAL_FEATURE_NAMES, retrieval_feature_vector(results)))
    return [float(feature_map.get(name, 0.0)) for name in feature_names]


def adaptive_guardrail(query, route_label, selected_mode, ltr_available, enabled_guardrails=None):
    """
    Apply lightweight runtime overrides on top of the learned easy/hard route.

    Returns a dict with override info or None when no guardrail should fire.
    """
    enabled = None
    if enabled_guardrails is not None:
        enabled = {str(name) for name in enabled_guardrails}

    def is_enabled(name):
        return enabled is None or name in enabled

    tokens = tokenize_query(query)
    ratio = stopword_ratio(tokens)
    prefix_like = is_prefix_like_query(tokens)
    non_stop_terms = [t for t in tokens if t not in STOPWORDS]

    if route_label == "hard":
        # Incomplete question-style prefixes are better served by lexical baseline.
        if (
            tokens
            and tokens[0] in QUESTION_WORDS
            and ratio >= 0.34
            and len(non_stop_terms) <= 1
            and is_enabled("hard_question_prefix_baseline")
        ):
            return {
                "selected_mode": "baseline",
                "route_guardrail": "hard_question_prefix_baseline",
            }

        # Longer mixed-intent informational queries benefit more from LTR.
        if (
            ltr_available
            and not prefix_like
            and len(non_stop_terms) >= 5
            and 0.25 <= ratio <= 0.40
            and is_enabled("hard_long_mix_ltr")
        ):
            return {
                "selected_mode": "ltr",
                "route_guardrail": "hard_long_mix_ltr",
            }

    if route_label == "easy" and selected_mode == "baseline" and ltr_available:
        # Topical multi-term easy queries are usually too broad for pure ES ranking.
        if (
            len(non_stop_terms) >= 3
            and ratio <= 0.34
            and not prefix_like
            and is_enabled("topical_easy_ltr")
        ):
            return {
                "selected_mode": "ltr",
                "route_guardrail": "topical_easy_ltr",
            }

    return None


def expert_to_mode(expert_name, default_mode):
    return EXPERT_TO_MODE.get(expert_name, default_mode)


def parse_hard_topk_policy(raw):
    """
    Parse hard top-k policy to sorted (delta_bound, topk) pairs.

    Supported formats:
    - str: "0.08:30,0.10:20,1.00:30"
    - list of [bound, topk] / (bound, topk) / {"delta":..., "top_k":...}
    """
    if not raw:
        return []

    pairs = []
    if isinstance(raw, str):
        for part in raw.split(","):
            item = part.strip()
            if not item:
                continue
            if ":" not in item:
                continue
            delta_s, topk_s = item.split(":", 1)
            try:
                delta = float(delta_s.strip())
                topk = int(topk_s.strip())
            except (TypeError, ValueError):
                continue
            if delta < 0 or topk <= 0:
                continue
            pairs.append((delta, topk))
    elif isinstance(raw, list):
        for item in raw:
            delta = None
            topk = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                delta, topk = item[0], item[1]
            elif isinstance(item, dict):
                delta = item.get("delta")
                topk = item.get("top_k", item.get("topk"))
            try:
                delta = float(delta)
                topk = int(topk)
            except (TypeError, ValueError):
                continue
            if delta < 0 or topk <= 0:
                continue
            pairs.append((delta, topk))

    pairs = sorted(pairs, key=lambda x: x[0])
    return pairs


class QueryRouter:
    """
    轻量查询路由器:
    - 优先加载模型 (LogReg + StandardScaler)
    - 模型不可用时使用启发式 fallback
    """

    def __init__(
        self,
        model_path="models/query_router.pkl",
        default_easy_mode="ltr",
        default_hard_mode="cross_encoder",
        default_hard_top_k=30,
    ):
        self.model_path = model_path
        self.default_easy_mode = default_easy_mode
        self.default_hard_mode = default_hard_mode
        self.default_hard_top_k = int(default_hard_top_k)
        self.model = None
        self.scaler = None
        self.hard_threshold = 0.5
        self.hard_top_k = self.default_hard_top_k
        self.hard_topk_policy = []
        self.easy_mode = default_easy_mode
        self.hard_mode = default_hard_mode
        self.loaded = False
        self.load_error = None
        self.model_meta = {}
        self.feature_names = FEATURE_NAMES
        self._try_load_model()

    def _try_load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            self.loaded = False
            return
        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)
            self.model = payload.get("model")
            self.scaler = payload.get("scaler")
            self.hard_threshold = float(payload.get("hard_threshold", 0.5))
            self.hard_top_k = int(payload.get("hard_top_k", self.default_hard_top_k))
            self.hard_topk_policy = parse_hard_topk_policy(payload.get("hard_topk_policy"))
            self.easy_mode = payload.get("easy_mode", self.default_easy_mode)
            self.hard_mode = payload.get("hard_mode", self.default_hard_mode)
            self.model_meta = payload.get("meta", {})
            self.feature_names = payload.get("feature_names") or FEATURE_NAMES
            self.loaded = self.model is not None and self.scaler is not None
        except Exception as e:
            self.loaded = False
            self.load_error = str(e)

    def _resolve_hard_top_k(self, hard_prob):
        base_top_k = max(1, int(self.hard_top_k))
        if not self.hard_topk_policy:
            return base_top_k
        delta = max(0.0, float(hard_prob) - float(self.hard_threshold))
        for bound, top_k in self.hard_topk_policy:
            if delta <= float(bound):
                return max(1, int(top_k))
        return base_top_k

    def _heuristic_route(self, query):
        tokens = tokenize_query(query)
        terms = len(tokens)
        chars = len(query or "")
        prefix_like = is_prefix_like_query(tokens)
        ratio = stopword_ratio(tokens)
        # 简单复杂度打分：越长、词数越多、停用词占比越低越偏 hard
        complexity = 0.0
        complexity += min(1.0, terms / 6.0)
        complexity += min(1.0, chars / 30.0)
        complexity += max(0.0, 1.0 - ratio)
        complexity -= 0.35 if prefix_like else 0.0
        hard_score = max(0.0, min(1.0, complexity / 2.5))
        if hard_score >= 0.5:
            return {
                "route_label": "hard",
                "route_confidence": hard_score,
                "route_source": "heuristic",
                "selected_mode": self.hard_mode,
                "hard_top_k": self._resolve_hard_top_k(hard_score),
            }
        return {
            "route_label": "easy",
            "route_confidence": 1.0 - hard_score,
            "route_source": "heuristic",
            "selected_mode": self.easy_mode,
            "hard_top_k": self.hard_top_k,
        }

    def route(self, query, results=None):
        if not self.loaded:
            return self._heuristic_route(query)

        try:
            feats = [router_feature_vector(query, results=results, feature_names=self.feature_names)]
            feats_scaled = self.scaler.transform(feats)
            prob = self.model.predict_proba(feats_scaled)[0]
            hard_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            if hard_prob >= self.hard_threshold:
                return {
                    "route_label": "hard",
                    "route_confidence": hard_prob,
                    "route_source": "model",
                    "selected_mode": self.hard_mode,
                    "hard_top_k": self._resolve_hard_top_k(hard_prob),
                }
            return {
                "route_label": "easy",
                "route_confidence": 1.0 - hard_prob,
                "route_source": "model",
                "selected_mode": self.easy_mode,
                "hard_top_k": self.hard_top_k,
            }
        except Exception:
            return self._heuristic_route(query)
