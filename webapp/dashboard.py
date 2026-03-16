def group_events_by_session(events_data):
    sessions = {}
    for event in events_data:
        session_id = (
            event.get("sessionId")
            or event.get("searchId")
            or event.get("search_id")
            or "unknown"
        )
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(event)
    return sessions


def render_research_dashboard_html(
    summary,
    sessions,
    *,
    ltr_available,
    router_loaded,
    feature_count,
):
    latency_stats = summary.get("latency_stats", {})
    adaptive_stats = summary.get("adaptive_stats", {})

    dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Dashboard - LTR Enhanced</title>
            <link rel="preconnect" href="https://fonts.googleapis.com" />
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
            <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&display=swap" rel="stylesheet" />
            <style>
                :root {{
                    --bg-1: #fff6e9;
                    --bg-2: #e6f5ef;
                    --bg-3: #d9ecf7;
                    --surface: #fffdfa;
                    --surface-soft: #f8f4ed;
                    --ink: #1f2a37;
                    --ink-muted: #5f6a75;
                    --line: #d7d8d2;
                    --primary: #0b6e4f;
                    --primary-soft: #e6f7ef;
                    --radius-xl: 22px;
                    --radius-lg: 16px;
                    --radius-md: 12px;
                    --shadow-lg: 0 20px 45px rgba(31, 42, 55, 0.12);
                    --shadow-md: 0 10px 24px rgba(31, 42, 55, 0.1);
                }}
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    min-height: 100vh;
                    color: var(--ink);
                    font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
                    background:
                        radial-gradient(1100px 550px at -10% -10%, rgba(11, 110, 79, 0.2), transparent 55%),
                        radial-gradient(900px 500px at 110% 10%, rgba(217, 119, 6, 0.22), transparent 50%),
                        linear-gradient(145deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
                    padding: 24px 16px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    border: 1px solid rgba(31, 42, 55, 0.08);
                    border-radius: var(--radius-xl);
                    padding: 24px;
                    background: linear-gradient(160deg, #fffefc 0%, var(--surface) 100%);
                    box-shadow: var(--shadow-lg);
                }}
                .header {{
                    border: 1px solid var(--line);
                    border-radius: var(--radius-lg);
                    padding: 20px;
                    margin-bottom: 18px;
                    background: linear-gradient(145deg, #ffffff 0%, var(--surface-soft) 100%);
                    box-shadow: var(--shadow-md);
                }}
                .eyebrow {{
                    margin: 0 0 6px;
                    font-size: 12px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.14em;
                    color: var(--primary);
                }}
                h1 {{
                    margin: 0;
                    font-family: "Source Serif 4", Georgia, serif;
                    font-size: clamp(1.7rem, 2.8vw, 2.4rem);
                    letter-spacing: -0.02em;
                }}
                .header-sub {{
                    margin: 8px 0 0;
                    color: var(--ink-muted);
                    font-size: 0.94rem;
                }}
                .ltr-badge {{
                    display: inline-block;
                    margin-top: 10px;
                    border-radius: 999px;
                    border: 1px solid rgba(11, 110, 79, 0.25);
                    background: var(--primary-soft);
                    color: var(--primary);
                    font-weight: 700;
                    font-size: 0.78rem;
                    padding: 4px 10px;
                }}
                .summary {{
                    background: white;
                    border: 1px solid var(--line);
                    border-radius: var(--radius-lg);
                    box-shadow: var(--shadow-md);
                    padding: 18px;
                    margin-bottom: 16px;
                }}
                h2 {{
                    margin: 0;
                    font-size: 1.15rem;
                }}
                .summary-head {{
                    display: flex;
                    justify-content: space-between;
                    gap: 12px;
                    align-items: flex-start;
                    flex-wrap: wrap;
                }}
                .summary-sub {{
                    margin: 6px 0 0;
                    color: var(--ink-muted);
                    font-size: 0.84rem;
                }}
                .filter-tabs {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    flex-wrap: wrap;
                }}
                .filter-tab {{
                    border-radius: 999px;
                    border: 1px solid rgba(11, 110, 79, 0.22);
                    background: #ffffff;
                    color: var(--primary);
                    font-size: 0.78rem;
                    font-weight: 700;
                    padding: 5px 11px;
                    cursor: pointer;
                }}
                .filter-tab.active {{
                    background: var(--primary);
                    color: #ffffff;
                    border-color: var(--primary);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 12px;
                    margin-top: 14px;
                }}
                .stat-card {{
                    border: 1px solid #dfe4df;
                    background: linear-gradient(145deg, #fff 0%, #fbf8f2 100%);
                    border-radius: var(--radius-md);
                    padding: 12px;
                    min-height: 90px;
                }}
                .stat-card.is-hidden {{
                    display: none;
                }}
                .stat-value {{
                    font-size: 1.55rem;
                    font-weight: 700;
                    color: var(--primary);
                    line-height: 1.15;
                }}
                .stat-label {{
                    color: var(--ink-muted);
                    font-size: 0.82rem;
                    margin-top: 6px;
                }}
                .session {{
                    border: 1px solid #dfe4df;
                    border-radius: var(--radius-md);
                    background: linear-gradient(145deg, #fff 0%, #fbf8f2 100%);
                    box-shadow: 0 8px 18px rgba(31, 42, 55, 0.08);
                    margin: 10px 0;
                    padding: 14px;
                }}
                .session h3 {{
                    margin: 0 0 6px;
                    font-size: 1rem;
                }}
                .session-meta {{
                    margin: 0 0 8px;
                    color: var(--ink-muted);
                    font-size: 0.84rem;
                }}
                .event {{
                    margin-top: 8px;
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #e5e7eb;
                    background: rgba(248, 250, 252, 0.8);
                    font-size: 0.86rem;
                    line-height: 1.5;
                }}
                .event-head {{
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    flex-wrap: wrap;
                    margin-bottom: 4px;
                }}
                .event-type {{
                    display: inline-block;
                    border-radius: 999px;
                    border: 1px solid rgba(11, 110, 79, 0.25);
                    background: var(--primary-soft);
                    color: var(--primary);
                    font-size: 0.74rem;
                    font-weight: 700;
                    padding: 3px 8px;
                }}
                .event-time {{
                    color: #7c8692;
                    font-size: 0.75rem;
                }}
                @media (max-width: 720px) {{
                    body {{ padding: 10px; }}
                    .container {{ padding: 14px; }}
                    .header {{ padding: 14px; }}
                    .summary-head {{ flex-direction: column; }}
                    .stats-grid {{ grid-template-columns: 1fr 1fr; }}
                }}
                @media (max-width: 520px) {{
                    .stats-grid {{ grid-template-columns: 1fr; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <p class="eyebrow">Adaptive Retrieval Lab</p>
                    <h1>Search Research Dashboard</h1>
                    <p class="header-sub">Real-time monitoring of routing, ranking quality proxies, and latency behavior.</p>
                    <span class="ltr-badge">LTR Enabled</span>
                </div>
                
                <div class="summary">
                    <div class="summary-head">
                        <div>
                            <h2>System Statistics</h2>
                            <p class="summary-sub">Toggle cards by metric family to focus on quality, latency, routing, or model status.</p>
                        </div>
                        <div class="filter-tabs" id="stat-filters">
                            <button class="filter-tab active" type="button" data-filter="all">All</button>
                            <button class="filter-tab" type="button" data-filter="quality">Quality</button>
                            <button class="filter-tab" type="button" data-filter="latency">Latency</button>
                            <button class="filter-tab" type="button" data-filter="routing">Routing</button>
                            <button class="filter-tab" type="button" data-filter="model">Model</button>
                        </div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-card" data-group="overview">
                            <div class="stat-value">{len(sessions)}</div>
                            <div class="stat-label">Total Sessions</div>
                        </div>
                        <div class="stat-card" data-group="overview">
                            <div class="stat-value">{summary['total_events']}</div>
                            <div class="stat-label">Total Events</div>
                        </div>
                        <div class="stat-card" data-group="overview quality">
                            <div class="stat-value">{summary['query_stats']['total_queries']}</div>
                            <div class="stat-label">Total Queries</div>
                        </div>
                        <div class="stat-card" data-group="quality">
                            <div class="stat-value">{summary['interaction_stats']['total_clicks']}</div>
                            <div class="stat-label">Result Clicks</div>
                        </div>
                        <div class="stat-card" data-group="quality">
                            <div class="stat-value">{summary['feedback_stats']['ctr_at_3'] * 100:.1f}%</div>
                            <div class="stat-label">CTR@3</div>
                        </div>
                        <div class="stat-card" data-group="quality">
                            <div class="stat-value">{summary['feedback_stats']['avg_click_rank']:.2f}</div>
                            <div class="stat-label">Avg Click Rank</div>
                        </div>
                        <div class="stat-card" data-group="quality">
                            <div class="stat-value">{summary['feedback_stats']['abandonment_rate'] * 100:.1f}%</div>
                            <div class="stat-label">Abandonment Rate</div>
                        </div>
                        <div class="stat-card" data-group="latency">
                            <div class="stat-value">{latency_stats.get('avg_total_ms', 0.0):.1f}</div>
                            <div class="stat-label">Avg Latency (ms)</div>
                        </div>
                        <div class="stat-card" data-group="latency">
                            <div class="stat-value">{latency_stats.get('p95_total_ms', 0.0):.1f}</div>
                            <div class="stat-label">P95 Latency (ms)</div>
                        </div>
                        <div class="stat-card" data-group="routing">
                            <div class="stat-value">{adaptive_stats.get('hard_rate', 0.0) * 100:.1f}%</div>
                            <div class="stat-label">Adaptive Hard Rate</div>
                        </div>
                        <div class="stat-card" data-group="model">
                            <div class="stat-value">{'✓' if ltr_available else '✗'}</div>
                            <div class="stat-label">LTR Model Status</div>
                        </div>
                        <div class="stat-card" data-group="routing model">
                            <div class="stat-value">{'✓' if router_loaded else 'Heuristic'}</div>
                            <div class="stat-label">Router Status</div>
                        </div>
                        <div class="stat-card" data-group="model">
                            <div class="stat-value">{feature_count}</div>
                            <div class="stat-label">Feature Dimensions</div>
                        </div>
                    </div>
                </div>
                
                <div class="summary">
                    <h2>Recent Sessions</h2>
        """

    for session_id, session_events in list(sessions.items())[:10]:
        dashboard_html += f"""
            <div class="session">
                <h3>Session: {session_id[:16]}...</h3>
                <p class="session-meta">Events: {len(session_events)}</p>
        """

        for event in session_events[-5:]:
            event_type = event.get("type") or event.get("event") or "unknown"
            timestamp = event.get("timestamp") or event.get("asctime") or "no timestamp"
            query = event.get("query", "")
            ranking_method = event.get("rankingMethod", "N/A")

            dashboard_html += f"""
                <div class="event">
                    <div class="event-head">
                        <span class="event-type">{event_type}</span>
                        <span class="event-time">{timestamp}</span>
                    </div>
                    {f'<br><strong>Query:</strong> {query}' if query else ''}
                    {f'<br><strong>Ranking:</strong> {ranking_method}' if ranking_method != 'N/A' else ''}
                </div>
            """

        dashboard_html += "</div>"

    dashboard_html += """
                </div>
            </div>
            <script>
                (() => {
                    const buttons = document.querySelectorAll('.filter-tab');
                    const cards = document.querySelectorAll('.stat-card');
                    if (!buttons.length || !cards.length) return;

                    const applyFilter = (selected) => {
                        cards.forEach((card) => {
                            const groups = (card.dataset.group || '').split(/\\s+/).filter(Boolean);
                            const visible = selected === 'all' || groups.includes(selected);
                            card.classList.toggle('is-hidden', !visible);
                        });
                        buttons.forEach((btn) => {
                            btn.classList.toggle('active', btn.dataset.filter === selected);
                        });
                    };

                    buttons.forEach((btn) => {
                        btn.addEventListener('click', () => applyFilter(btn.dataset.filter || 'all'));
                    });
                })();
            </script>
        </body>
        </html>
    """

    return dashboard_html
