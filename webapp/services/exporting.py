from datetime import datetime

from ..analytics import build_event_summary, load_events_from_log
from ..dashboard import group_events_by_session, render_research_dashboard_html


def filter_events_by_session(events_data, session_id_filter):
    if not session_id_filter:
        return events_data
    return [
        event for event in events_data
        if isinstance(event, dict) and event.get("sessionId") == session_id_filter
    ]


def build_export_payload(
    log_file,
    *,
    session_id_filter="",
    load_events_fn=load_events_from_log,
    summary_fn=build_event_summary,
    now_factory=datetime.now,
):
    events_data = load_events_fn(log_file)
    events_data = filter_events_by_session(events_data, session_id_filter)
    summary = summary_fn(events_data)
    return {
        "export_timestamp": now_factory().isoformat(),
        "session_id_filter": session_id_filter or None,
        "summary": summary,
        "raw_events": events_data,
    }


def build_dashboard_html(
    log_file,
    *,
    ltr_available,
    router_loaded,
    feature_count,
    load_events_fn=load_events_from_log,
    summary_fn=build_event_summary,
    session_group_fn=group_events_by_session,
    render_fn=render_research_dashboard_html,
):
    events_data = load_events_fn(log_file, limit=100)
    summary = summary_fn(events_data)
    sessions = session_group_fn(events_data)
    return render_fn(
        summary,
        sessions,
        ltr_available=ltr_available,
        router_loaded=router_loaded,
        feature_count=feature_count,
    )
