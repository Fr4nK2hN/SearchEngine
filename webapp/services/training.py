import os


def train_ltr_ranker(
    es,
    cross_encoder_model,
    feature_extractor,
    *,
    model_path,
    num_queries=50,
    n_estimators=200,
    learning_rate=0.05,
    ranker_cls=None,
    generator_cls=None,
):
    if generator_cls is None:
        from ranking.training_data_generator import TrainingDataGenerator

        generator_cls = TrainingDataGenerator
    if ranker_cls is None:
        from ranking.ranker import LTRRanker

        ranker_cls = LTRRanker

    generator = generator_cls(es, cross_encoder_model)
    queries = generator.generate_training_queries(num_queries=num_queries)
    training_data = generator.generate_training_data(queries, docs_per_query=30)

    split_idx = int(len(training_data) * 0.8)
    train_set = training_data[:split_idx]
    val_set = training_data[split_idx:]

    ltr_ranker = ranker_cls(feature_extractor)
    ltr_ranker.train(
        training_data=train_set,
        validation_data=val_set,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )

    model_dir = os.path.dirname(model_path) or "."
    os.makedirs(model_dir, exist_ok=True)
    ltr_ranker.save_model(model_path)

    return ltr_ranker, len(train_set), len(val_set)
