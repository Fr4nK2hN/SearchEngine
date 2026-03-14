import json


def test_build_idf_accepts_bare_output_filename(monkeypatch, tmp_path):
    from tools.build_idf import build_idf

    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"title": "alpha beta", "content": "alpha beta gamma"},
                {"title": "beta gamma", "content": "gamma delta"},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    idf_dict = build_idf(str(corpus_path), "idf.json")

    assert (tmp_path / "idf.json").exists()
    assert idf_dict


def test_feedback_builder_prefers_server_result_ids_and_route_metadata(monkeypatch):
    from tools import build_feedback_ltr_data as mod

    def fail_search(*args, **kwargs):
        raise AssertionError("live search replay should not run when server result ids exist")

    def fake_fetch_documents(base_url, doc_ids, doc_scores=None, timeout=20):
        assert base_url == "http://unused"
        assert doc_ids == ["doc-1"]
        assert doc_scores == [1.5]
        return [
            {
                "_id": "doc-1",
                "_score": 1.5,
                "_source": {
                    "title": "Alpha",
                    "content": "Alpha content",
                    "related_queries": ["alpha query"],
                },
            }
        ]

    monkeypatch.setattr(mod, "fetch_search_results", fail_search)
    monkeypatch.setattr(mod, "fetch_documents_by_ids", fake_fetch_documents)

    events = [
        {
            "type": "query_submitted",
            "searchId": "s1",
            "query": "alpha",
            "mode": "adaptive",
            "route_selected_mode": "hybrid",
            "route_rerank_top_n": 30,
        },
        {
            "type": "search_completed",
            "searchId": "s1",
            "sessionId": "sess-1",
            "query": "alpha",
            "mode": "adaptive",
            "result_ids": ["doc-1"],
            "result_scores": [1.5],
        },
        {
            "type": "serp_impression",
            "searchId": "s1",
            "results": ["doc-1"],
            "result_scores": [1.5],
        },
        {
            "type": "result_click_confirmed",
            "searchId": "s1",
            "doc_id": "doc-1",
            "rank": 2,
        },
    ]

    samples, dropped, stats = mod.build_from_feedback(
        events=events,
        trace_by_search_id={},
        base_url="http://unused",
        candidate_source="search",
        fallback_mode="baseline",
        use_confirmed_only=True,
        min_clicks_per_search=1,
    )

    assert not dropped
    assert stats["search_count"] == 1
    assert len(samples) == 1
    assert samples[0]["meta"]["mode"] == "hybrid"
    assert samples[0]["meta"]["requested_mode"] == "adaptive"
    assert samples[0]["meta"]["candidate_source"] == "server_result_ids_api"
    assert samples[0]["meta"]["route_rerank_top_n"] == 30
    assert samples[0]["relevance_labels"] == [3]


def test_feedback_builder_replays_selected_mode_for_adaptive_search(monkeypatch):
    from tools import build_feedback_ltr_data as mod

    calls = {}

    def fake_fetch(base_url, query, mode, rerank_top_n=None, timeout=20):
        calls["base_url"] = base_url
        calls["query"] = query
        calls["mode"] = mode
        calls["rerank_top_n"] = rerank_top_n
        return [
            {
                "_id": "doc-9",
                "_score": 0.9,
                "_source": {
                    "title": "Replay Doc",
                    "content": "Replay content",
                    "related_queries": [],
                },
            }
        ]

    monkeypatch.setattr(mod, "fetch_search_results", fake_fetch)

    events = [
        {
            "type": "query_submitted",
            "searchId": "s2",
            "query": "beta",
            "mode": "adaptive",
            "route_selected_mode": "cross_encoder",
            "route_rerank_top_n": 20,
        },
        {
            "type": "search_completed",
            "searchId": "s2",
            "sessionId": "sess-2",
            "query": "beta",
            "mode": "adaptive",
        },
        {
            "type": "result_click_confirmed",
            "searchId": "s2",
            "doc_id": "doc-9",
            "rank": 1,
        },
    ]

    samples, dropped, _ = mod.build_from_feedback(
        events=events,
        trace_by_search_id={},
        base_url="http://example.test",
        candidate_source="search",
        fallback_mode="baseline",
        use_confirmed_only=True,
        min_clicks_per_search=1,
    )

    assert not dropped
    assert calls == {
        "base_url": "http://example.test",
        "query": "beta",
        "mode": "cross_encoder",
        "rerank_top_n": 20,
    }
    assert samples[0]["meta"]["mode"] == "cross_encoder"
    assert samples[0]["meta"]["requested_mode"] == "adaptive"


def test_feedback_builder_rebuilds_from_trace_using_server_result_ids():
    from tools import build_feedback_ltr_data as mod

    events = [
        {
            "type": "query_submitted",
            "searchId": "s3",
            "query": "gamma",
            "mode": "baseline",
            "result_ids": ["doc-2", "doc-1"],
            "result_scores": [2.0, 1.0],
        },
        {
            "type": "result_click_confirmed",
            "searchId": "s3",
            "doc_id": "doc-1",
            "rank": 2,
        },
    ]

    trace_hits = [
        {
            "_id": "doc-1",
            "_score": 0.5,
            "_source": {"title": "One", "content": "First", "related_queries": []},
        },
        {
            "_id": "doc-2",
            "_score": 0.4,
            "_source": {"title": "Two", "content": "Second", "related_queries": []},
        },
    ]

    samples, dropped, _ = mod.build_from_feedback(
        events=events,
        trace_by_search_id={"s3": trace_hits},
        base_url="",
        candidate_source="trace",
        fallback_mode="baseline",
        use_confirmed_only=True,
        min_clicks_per_search=1,
    )

    assert not dropped
    assert samples[0]["meta"]["candidate_source"] == "server_result_ids_trace"
    assert [doc["id"] for doc in samples[0]["documents"]] == ["doc-2", "doc-1"]
    assert [doc["es_score"] for doc in samples[0]["documents"]] == [2.0, 1.0]
