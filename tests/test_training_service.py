import tempfile
import unittest

from webapp.services.training import train_ltr_ranker


class FakeGenerator:
    def __init__(self, es, cross_encoder_model):
        self.es = es
        self.cross_encoder_model = cross_encoder_model

    def generate_training_queries(self, num_queries):
        return [f"q{idx}" for idx in range(num_queries)]

    def generate_training_data(self, queries, docs_per_query):
        return [{"query": query, "docs_per_query": docs_per_query} for query in queries]


class FakeRanker:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.train_calls = []
        self.saved_paths = []

    def train(self, **kwargs):
        self.train_calls.append(kwargs)
        return self

    def save_model(self, path):
        self.saved_paths.append(path)
        return path


class TrainingServiceTests(unittest.TestCase):
    def test_train_ltr_ranker_splits_data_and_saves_model(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_path = f"{tempdir}/ltr_model.pkl"
            feature_extractor = object()

            ranker, train_count, val_count = train_ltr_ranker(
                es=object(),
                cross_encoder_model=object(),
                feature_extractor=feature_extractor,
                model_path=model_path,
                num_queries=5,
                n_estimators=300,
                learning_rate=0.1,
                ranker_cls=FakeRanker,
                generator_cls=FakeGenerator,
            )

        self.assertIsInstance(ranker, FakeRanker)
        self.assertIs(ranker.feature_extractor, feature_extractor)
        self.assertEqual(train_count, 4)
        self.assertEqual(val_count, 1)
        self.assertEqual(ranker.saved_paths, [model_path])
        self.assertEqual(ranker.train_calls[0]["n_estimators"], 300)
        self.assertEqual(ranker.train_calls[0]["learning_rate"], 0.1)
        self.assertEqual(len(ranker.train_calls[0]["training_data"]), 4)
        self.assertEqual(len(ranker.train_calls[0]["validation_data"]), 1)


if __name__ == "__main__":
    unittest.main()
