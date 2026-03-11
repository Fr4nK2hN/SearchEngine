import os
import json
import argparse
import random
from ranking.feature_extractor import FeatureExtractor
from ranking.evaluator import RankingEvaluator
CrossEncoder = None

class BaselineRanker:
    def predict(self, query, documents):
        return [float(doc.get('es_score', 0.0)) for doc in documents]

class CrossEncoderRanker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        global CrossEncoder
        if CrossEncoder is None:
            try:
                from sentence_transformers import CrossEncoder as CE
                CrossEncoder = CE
            except Exception as e:
                raise RuntimeError(str(e))
        self.model = CrossEncoder(model_name)
    def predict(self, query, documents):
        passages = [doc.get('content', '') for doc in documents]
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

class HybridRanker:
    def __init__(self, ltr_ranker, cross_ranker):
        self.ltr = ltr_ranker
        self.cross = cross_ranker
    def predict(self, query, documents):
        ltr_scores = self.ltr.predict(query, documents)
        idx_sorted = sorted(range(len(documents)), key=lambda i: ltr_scores[i], reverse=True)
        top_idx = idx_sorted[:20]
        top_docs = [documents[i] for i in top_idx]
        cross_scores = self.cross.predict(query, top_docs)
        final_scores = list(ltr_scores)
        for j, i in enumerate(top_idx):
            final_scores[i] = 0.6 * float(ltr_scores[i]) + 0.4 * float(cross_scores[j])
        return final_scores

def load_test_data(path, n=None, sample_method='tail', seed=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    total = len(data)
    if n is None:
        n = max(10, int(total * 0.2))
    n = max(1, min(n, total))
    if sample_method == 'random':
        if seed is not None:
            random.seed(seed)
        idx = random.sample(range(total), n)
        idx.sort()
        return [data[i] for i in idx]
    return data[-n:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ltr_training_data.json')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--sample', choices=['tail', 'random'], default='tail')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--k', type=int, nargs='*', default=[3, 5, 10])
    parser.add_argument('--no_cross', action='store_true')
    parser.add_argument('--save_plot', default='models/offline_eval_comparison.png')
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        print('测试集未找到: ' + data_path)
        return
    test_set = load_test_data(data_path, n=args.n, sample_method=args.sample, seed=args.seed)
    feature_extractor = FeatureExtractor()
    evaluator = RankingEvaluator()
    rankers = {}
    rankers['Baseline'] = BaselineRanker()
    ltr_model_path = 'models/ltr_model.pkl'
    ltr_available = os.path.exists(ltr_model_path)
    ltr = None
    if ltr_available:
        try:
            from ranking.ranker import LTRRanker
            ltr = LTRRanker(feature_extractor)
            ltr.load_model(ltr_model_path)
            rankers['LTR'] = ltr
        except Exception as e:
            print('LTR unavailable due to dependency error: ' + str(e))
            ltr_available = False
    cross = None
    if not args.no_cross:
        try:
            cross = CrossEncoderRanker()
            rankers['Cross-Encoder'] = cross
            if ltr_available and ltr is not None:
                rankers['Hybrid'] = HybridRanker(ltr, cross)
        except Exception:
            pass
    comparison = evaluator.compare_rankers(test_set, rankers, k_values=args.k)
    for name, res in comparison.items():
        print(name)
        evaluator.print_results(res)
    try:
        evaluator.plot_comparison(comparison, save_path=args.save_plot)
    except Exception:
        pass

if __name__ == '__main__':
    main()
