from __future__ import annotations

from html import escape
import json
from typing import Any


def render_taste_os_page_html(service: Any) -> str:
    config = {
        "modelName": service.model_name,
        "modelType": service.model_type,
        "runDir": str(service.run_dir),
        "outputDir": str(service.output_dir),
        "maxTopK": int(service.max_top_k),
        "defaultTopK": min(5, int(service.max_top_k)),
        "requiresAuth": bool(service.auth_token),
    }
    config_json = json.dumps(config)
    model_label = escape(f"{service.model_name} [{service.model_type}]")
    run_dir = escape(str(service.run_dir))
    output_dir = escape(str(service.output_dir))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Taste OS Session Studio</title>
  <style>
    :root {{
      --paper: #f4efe5;
      --paper-strong: #fbf8f1;
      --ink: #1e1d1a;
      --muted: #625b4f;
      --line: rgba(30, 29, 26, 0.12);
      --accent: #b64f2f;
      --accent-soft: rgba(182, 79, 47, 0.12);
      --accent-deep: #8e3419;
      --olive: #687347;
      --olive-soft: rgba(104, 115, 71, 0.12);
      --sky: #cadfd9;
      --card-shadow: 0 18px 48px rgba(58, 46, 36, 0.10);
      --mono: "IBM Plex Mono", "SFMono-Regular", "Menlo", monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: "Avenir Next", "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182, 79, 47, 0.18), transparent 28%),
        radial-gradient(circle at right 12%, rgba(104, 115, 71, 0.16), transparent 22%),
        linear-gradient(180deg, #f7f2e8 0%, #efe7d9 54%, #ece2d2 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1320px, calc(100vw - 32px));
      margin: 24px auto 40px;
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
      gap: 20px;
    }}
    .panel, .stage {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: rgba(251, 248, 241, 0.88);
      box-shadow: var(--card-shadow);
      backdrop-filter: blur(12px);
    }}
    .panel {{
      padding: 24px;
      position: sticky;
      top: 24px;
      align-self: start;
    }}
    .stage {{
      padding: 26px;
      overflow: hidden;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-deep);
      font: 600 12px/1 var(--mono);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1, h2, h3 {{
      font-family: var(--serif);
      font-weight: 700;
      margin: 0;
    }}
    h1 {{
      font-size: clamp(2.3rem, 4vw, 4rem);
      line-height: 0.96;
      margin-top: 14px;
      max-width: 10ch;
    }}
    h2 {{
      font-size: 1.55rem;
      margin-bottom: 10px;
    }}
    p, li, label, input, textarea, button {{
      font-size: 0.98rem;
      line-height: 1.45;
    }}
    .lede {{
      margin: 14px 0 24px;
      color: var(--muted);
      max-width: 34ch;
    }}
    .meta {{
      display: grid;
      gap: 10px;
      margin-bottom: 24px;
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(202, 223, 217, 0.32), rgba(251, 248, 241, 0.92));
      border: 1px solid rgba(104, 115, 71, 0.14);
    }}
    .meta-row {{
      display: grid;
      gap: 4px;
    }}
    .meta-label {{
      color: var(--muted);
      font: 600 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .meta-value {{
      font-weight: 600;
      word-break: break-word;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .choices {{
      display: grid;
      gap: 10px;
    }}
    .choice-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }}
    .choice {{
      padding: 14px 14px 12px;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      text-align: left;
    }}
    .choice:hover {{ transform: translateY(-2px); }}
    .choice.active {{
      border-color: rgba(182, 79, 47, 0.55);
      background: linear-gradient(180deg, rgba(182, 79, 47, 0.12), rgba(255, 255, 255, 0.9));
    }}
    .choice strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 0.98rem;
    }}
    .choice span {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .field {{
      display: grid;
      gap: 7px;
    }}
    .field label {{
      font-weight: 600;
    }}
    input[type="number"], textarea {{
      width: 100%;
      border: 1px solid rgba(30, 29, 26, 0.15);
      border-radius: 16px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      font-family: var(--sans);
    }}
    textarea {{
      min-height: 92px;
      resize: vertical;
    }}
    .inline-options {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .inline-options label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    button.primary {{
      border: none;
      border-radius: 999px;
      padding: 13px 20px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%);
      color: white;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 14px 28px rgba(182, 79, 47, 0.20);
    }}
    button.primary:hover {{ transform: translateY(-1px); }}
    .hint {{
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .stage-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 22px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .summary-card {{
      padding: 16px;
      border-radius: 22px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(202, 223, 217, 0.22));
      animation: rise 300ms ease;
    }}
    .summary-card .label {{
      color: var(--muted);
      font: 600 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .summary-card .value {{
      font-family: var(--serif);
      font-size: 1.45rem;
      line-height: 1.0;
    }}
    .result-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .result-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.78);
      animation: rise 320ms ease;
    }}
    .result-card.wide {{
      grid-column: 1 / -1;
    }}
    .result-card ol, .result-card ul {{
      margin: 10px 0 0;
      padding-left: 18px;
    }}
    .history-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .mini-card {{
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.72);
      animation: rise 340ms ease;
    }}
    .mini-card ul, .mini-card ol {{
      margin: 10px 0 0;
      padding-left: 18px;
    }}
    .pill-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(104, 115, 71, 0.12);
      color: var(--olive);
      font: 600 12px/1 var(--mono);
    }}
    .pill.warn {{
      background: rgba(182, 79, 47, 0.12);
      color: var(--accent-deep);
    }}
    .candidate-list li, .plan-list li, .transcript-list li {{
      margin-bottom: 10px;
    }}
    .feedback-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .feedback-button {{
      border: 1px solid rgba(30, 29, 26, 0.12);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      cursor: pointer;
      font: 600 12px/1 var(--mono);
    }}
    .feedback-button:hover {{
      border-color: rgba(182, 79, 47, 0.35);
      color: var(--accent-deep);
    }}
    .metric-line {{
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .artifact-links a {{
      color: var(--accent-deep);
      text-decoration: none;
      font-weight: 600;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px;
      border-radius: 22px;
      border: 1px dashed rgba(30, 29, 26, 0.18);
      background: rgba(255, 255, 255, 0.54);
    }}
    .status {{
      min-height: 1.3em;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 0.82rem;
    }}
    .status.error {{
      color: #9b2314;
    }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 960px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .panel {{
        position: static;
      }}
      .result-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel">
      <div class="eyebrow">Taste OS Studio</div>
      <h1>Shape a listening session before it starts drifting.</h1>
      <p class="lede">This browser surface sits on top of the Taste OS planner, so we can steer a session, stress it with an event, and inspect the explanation, guardrails, and recovery path in one place.</p>
      <div class="meta">
        <div class="meta-row">
          <div class="meta-label">Serveable model</div>
          <div class="meta-value">{model_label}</div>
        </div>
        <div class="meta-row">
          <div class="meta-label">Run</div>
          <div class="meta-value">{run_dir}</div>
        </div>
        <div class="meta-row">
          <div class="meta-label">Artifact output</div>
          <div class="meta-value">{output_dir}</div>
        </div>
      </div>
      <div class="stack">
        <section class="choices">
          <h2>Mode</h2>
          <div id="mode-grid" class="choice-grid"></div>
        </section>
        <section class="choices">
          <h2>Scenario</h2>
          <div id="scenario-grid" class="choice-grid"></div>
        </section>
        <div class="field">
          <label for="top-k">Top candidates</label>
          <input id="top-k" type="number" min="1" max="{int(service.max_top_k)}" value="{min(5, int(service.max_top_k))}">
        </div>
        <div class="field">
          <label for="recent-artists">Recent artists</label>
          <textarea id="recent-artists" placeholder="Artist A|Artist B|Artist C"></textarea>
        </div>
        <div class="inline-options">
          <label><input id="include-video" type="checkbox"> Include video history</label>
          <label><input id="persist-artifacts" type="checkbox" checked> Persist artifacts</label>
          <label><input id="use-feedback-memory" type="checkbox" checked> Seed from feedback memory</label>
        </div>
        <div class="actions">
          <button id="run-session" class="primary" type="button">Generate Session</button>
          <button id="refresh-memory" type="button">Refresh Memory</button>
          <div class="hint">POSTs to <code>/taste-os/session</code>.</div>
        </div>
        <div id="status" class="status">Loading catalog…</div>
      </div>
    </aside>
    <main class="stage">
      <div class="stage-header">
        <div>
          <div class="eyebrow">Session Surface</div>
          <h2>Plan, rationale, and recovery flow</h2>
        </div>
        <div class="hint">Use this page to compare modes without dropping into raw JSON.</div>
      </div>
      <div id="summary-grid" class="summary-grid"></div>
      <div id="history-grid" class="history-grid"></div>
      <div id="result-grid" class="result-grid">
        <div class="empty">Choose a mode and scenario, then generate a session to see the opening choice, candidate stack, baseline plan, guardrails, and adaptive transcript.</div>
      </div>
    </main>
  </div>
  <script>
    const config = {config_json};
    const state = {{
      mode: "focus",
      scenario: "steady",
      catalog: {{ modes: [], scenarios: [] }},
      history: {{ recent_sessions: [], feedback_memory: {{}} }},
      payload: null,
    }};

    const modeGrid = document.getElementById("mode-grid");
    const scenarioGrid = document.getElementById("scenario-grid");
    const resultGrid = document.getElementById("result-grid");
    const summaryGrid = document.getElementById("summary-grid");
    const historyGrid = document.getElementById("history-grid");
    const statusNode = document.getElementById("status");
    const topKInput = document.getElementById("top-k");
    const recentArtistsInput = document.getElementById("recent-artists");
    const includeVideoInput = document.getElementById("include-video");
    const persistArtifactsInput = document.getElementById("persist-artifacts");
    const useFeedbackMemoryInput = document.getElementById("use-feedback-memory");
    const runButton = document.getElementById("run-session");
    const refreshMemoryButton = document.getElementById("refresh-memory");

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function setStatus(message, isError = false) {{
      statusNode.textContent = message;
      statusNode.className = isError ? "status error" : "status";
    }}

    function renderChoiceGrid(items, activeValue, container, key, descriptionKey) {{
      container.innerHTML = items.map((item) => `
        <button class="choice ${{
          item.name === activeValue ? "active" : ""
        }}" data-${{key}}="${{escapeHtml(item.name)}}">
          <strong>${{escapeHtml(item.name)}}</strong>
          <span>${{escapeHtml(item[descriptionKey])}}</span>
        </button>
      `).join("");
    }}

    function renderCatalog() {{
      renderChoiceGrid(state.catalog.modes, state.mode, modeGrid, "mode", "description");
      renderChoiceGrid(state.catalog.scenarios, state.scenario, scenarioGrid, "scenario", "description");
      modeGrid.querySelectorAll("[data-mode]").forEach((node) => {{
        node.addEventListener("click", () => {{
          state.mode = node.dataset.mode;
          renderCatalog();
        }});
      }});
      scenarioGrid.querySelectorAll("[data-scenario]").forEach((node) => {{
        node.addEventListener("click", () => {{
          state.scenario = node.dataset.scenario;
          renderCatalog();
        }});
      }});
    }}

    function summaryCards(payload) {{
      const summary = payload.demo_summary || {{}};
      const request = payload.request || {{}};
      const memory = payload.memory_summary || {{}};
      return [
        {{ label: "Mode", value: request.mode || "n/a" }},
        {{ label: "Scenario", value: request.scenario || "n/a" }},
        {{ label: "Top Artist", value: summary.top_artist || "n/a" }},
        {{ label: "Replans", value: String(summary.adaptive_replans ?? 0) }},
        {{ label: "Safe Route Steps", value: String(summary.adaptive_safe_route_steps ?? 0) }},
        {{ label: "Memory Seeds", value: String((memory.seed_artists || []).length) }},
      ];
    }}

    function renderSummary(payload) {{
      summaryGrid.innerHTML = summaryCards(payload).map((card) => `
        <div class="summary-card">
          <div class="label">${{escapeHtml(card.label)}}</div>
          <div class="value">${{escapeHtml(card.value)}}</div>
        </div>
      `).join("");
    }}

    function renderHistory(history) {{
      const feedback = history.feedback_memory || {{}};
      const recentSessions = Array.isArray(history.recent_sessions) ? history.recent_sessions : [];
      const topAffinities = Array.isArray(feedback.top_affinities) ? feedback.top_affinities : [];
      const avoidArtists = Array.isArray(feedback.avoid_artists) ? feedback.avoid_artists : [];
      const seedArtists = Array.isArray(feedback.seed_artists) ? feedback.seed_artists : [];
      historyGrid.innerHTML = `
        <section class="mini-card">
          <h3>Taste Memory</h3>
          <div class="metric-line">events=${{escapeHtml(feedback.event_count ?? 0)}} | artists=${{escapeHtml(feedback.artist_count ?? 0)}}</div>
          <div class="pill-list">
            ${{seedArtists.map((artist) => `<span class="pill">${{escapeHtml(artist)}}</span>`).join("") || '<span class="metric-line">No memory seeds yet.</span>'}}
          </div>
        </section>
        <section class="mini-card">
          <h3>Affinities</h3>
          <ul>
            ${{
              topAffinities.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">score=${{escapeHtml(row.net_score)}} | likes=${{escapeHtml(row.like_count)}} | repeats=${{escapeHtml(row.repeat_count)}}</div>
                </li>
              `).join("") || "<li>No positive feedback recorded yet.</li>"
            }}
          </ul>
        </section>
        <section class="mini-card">
          <h3>Avoid</h3>
          <ul>
            ${{
              avoidArtists.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">score=${{escapeHtml(row.net_score)}} | dislikes=${{escapeHtml(row.dislike_count)}} | skips=${{escapeHtml(row.skip_count)}}</div>
                </li>
              `).join("") || "<li>No avoid signals recorded yet.</li>"
            }}
          </ul>
        </section>
        <section class="mini-card">
          <h3>Recent Sessions</h3>
          <ol>
            ${{
              recentSessions.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.mode)}} / ${{escapeHtml(row.scenario)}}</strong>
                  <div class="metric-line">top=${{escapeHtml(row.top_artist || "n/a")}} | replans=${{escapeHtml(row.adaptive_replans ?? 0)}}</div>
                </li>
              `).join("") || "<li>No sessions recorded yet.</li>"
            }}
          </ol>
        </section>
      `;
    }}

    function renderResults(payload) {{
      const risk = payload.risk_summary || {{}};
      const fallback = payload.fallback_policy || {{}};
      const adaptive = payload.adaptive_session || {{}};
      const service = payload.service || {{}};
      const memory = payload.memory_summary || {{}};
      const topCandidates = Array.isArray(payload.top_candidates) ? payload.top_candidates : [];
      const journeyPlan = Array.isArray(payload.journey_plan) ? payload.journey_plan : [];
      const whyThisNext = Array.isArray(payload.why_this_next) ? payload.why_this_next : [];
      const transcript = Array.isArray(adaptive.transcript) ? adaptive.transcript : [];
      const effectiveRecentArtists = Array.isArray(memory.effective_recent_artists) ? memory.effective_recent_artists : [];
      const topAffinities = Array.isArray(memory.top_affinities) ? memory.top_affinities : [];
      const avoidArtists = Array.isArray(memory.avoid_artists) ? memory.avoid_artists : [];

      function feedbackButtons(artistName) {{
        return `
          <div class="feedback-buttons">
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="like">Like</button>
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="repeat">Repeat</button>
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="skip">Skip</button>
            <button class="feedback-button" type="button" data-feedback-artist="${{escapeHtml(artistName)}}" data-feedback-signal="dislike">Dislike</button>
          </div>
        `;
      }}

      const artifactBlock = service.persisted
        ? `<div class="artifact-links">
             <a href="${{escapeHtml(service.artifact_json_url || "#")}}" target="_blank" rel="noreferrer">Artifact JSON</a><div class="metric-line">${{escapeHtml(service.artifact_json || "")}}</div>
             <a href="${{escapeHtml(service.artifact_md_url || "#")}}" target="_blank" rel="noreferrer">Artifact MD</a><div class="metric-line">${{escapeHtml(service.artifact_md || "")}}</div>
           </div>`
        : `<div class="metric-line">Artifact persistence is off for this session.</div>`;

      resultGrid.innerHTML = `
        <section class="result-card">
          <h3>Why This Next</h3>
          <ul>${{whyThisNext.map((line) => `<li>${{escapeHtml(line)}}</li>`).join("") || "<li>No rationale available.</li>"}}</ul>
        </section>
        <section class="result-card">
          <h3>Guardrails</h3>
          <ul>
            <li>Risk state: <strong>${{escapeHtml(risk.risk_state || "n/a")}}</strong></li>
            <li>End risk: <strong>${{escapeHtml(risk.current_end_risk ?? "n/a")}}</strong></li>
            <li>Friction bucket: <strong>${{escapeHtml(risk.friction_bucket || "n/a")}}</strong></li>
            <li>Fallback policy: <strong>${{escapeHtml(fallback.active_policy_name || "n/a")}}</strong></li>
          </ul>
          <div class="metric-line">${{escapeHtml(fallback.reason || "")}}</div>
        </section>
        <section class="result-card wide">
          <h3>Top Candidates</h3>
          <ol class="candidate-list">
            ${{
              topCandidates.map((row) => `
                <li>
                  <strong>${{escapeHtml(row.rank)}}. ${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">
                    model_prob=${{escapeHtml(row.model_probability)}} |
                    surface=${{escapeHtml(row.surface_score)}} |
                    mode=${{escapeHtml(row.mode_score)}} |
                    continuity=${{escapeHtml(row.continuity)}} |
                    novelty=${{escapeHtml(row.novelty)}}
                  </div>
                  <div class="metric-line">
                    memory=${{escapeHtml(row.memory_state || "neutral")}} |
                    memory_score=${{escapeHtml(row.memory_net_score ?? 0)}}
                  </div>
                  ${{feedbackButtons(row.artist_name || "")}}
                </li>
              `).join("") || "<li>No candidates returned.</li>"
            }}
          </ol>
        </section>
        <section class="result-card">
          <h3>Baseline Plan</h3>
          <ol class="plan-list">
            ${{
              journeyPlan.map((row) => `
                <li>
                  <strong>Step ${{escapeHtml(row.step)}} -> ${{escapeHtml(row.artist_name)}}</strong>
                  <div class="metric-line">
                    transition=${{escapeHtml(row.transition_probability)}} |
                    mode_score=${{escapeHtml(row.mode_score)}}
                  </div>
                </li>
              `).join("") || "<li>No plan returned.</li>"
            }}
          </ol>
        </section>
        <section class="result-card">
          <h3>Adaptive Session</h3>
          <div class="metric-line">${{escapeHtml(adaptive.description || "")}}</div>
          <ol class="transcript-list">
            ${{
              transcript.map((row) => `
                <li>
                  <strong>Step ${{escapeHtml(row.step)}} -> ${{escapeHtml(row.selected_artist)}}</strong>
                  <div class="metric-line">
                    origin=${{escapeHtml(row.plan_origin)}} |
                    policy=${{escapeHtml(row.policy_name)}} |
                    end_risk=${{escapeHtml(row.end_risk)}}
                  </div>
                  ${{row.why_changed ? `<div class="metric-line">${{escapeHtml(row.why_changed)}}</div>` : ""}}
                  ${{row.event_applied_after_step ? `<div class="metric-line">event=${{escapeHtml(row.event_applied_after_step)}} | ${{escapeHtml(row.event_summary || "")}}</div>` : ""}}
                </li>
              `).join("") || "<li>No transcript available.</li>"
            }}
          </ol>
        </section>
        <section class="result-card wide">
          <h3>Service Output</h3>
          <div class="metric-line">Run dir: ${{escapeHtml(service.run_dir || config.runDir)}}</div>
          <div class="metric-line">Output dir: ${{escapeHtml(service.output_dir || config.outputDir)}}</div>
          <div class="metric-line">Session id: ${{escapeHtml(service.session_id || "n/a")}}</div>
          <div class="metric-line">Created at: ${{escapeHtml(service.created_at || "n/a")}}</div>
          ${{artifactBlock}}
        </section>
        <section class="result-card wide">
          <h3>Memory Context</h3>
          <div class="metric-line">Effective recent artists: ${{escapeHtml(effectiveRecentArtists.join(" | ") || "none")}}</div>
          <div class="pill-list">
            ${{topAffinities.slice(0, 4).map((row) => `<span class="pill">${{escapeHtml(row.artist_name)}} · ${{escapeHtml(row.net_score)}}</span>`).join("")}}
            ${{avoidArtists.slice(0, 3).map((row) => `<span class="pill warn">${{escapeHtml(row.artist_name)}} · ${{escapeHtml(row.net_score)}}</span>`).join("")}}
          </div>
        </section>
      `;

      resultGrid.querySelectorAll("[data-feedback-artist]").forEach((node) => {{
        node.addEventListener("click", async () => {{
          const artistName = node.dataset.feedbackArtist || "";
          const signal = node.dataset.feedbackSignal || "";
          await submitFeedback(service.session_id || "", artistName, signal);
        }});
      }});
    }}

    async function loadCatalog() {{
      const response = await fetch("/taste-os/catalog");
      if (!response.ok) {{
        throw new Error(`Catalog request failed with status ${{response.status}}`);
      }}
      state.catalog = await response.json();
      renderCatalog();
      setStatus("Catalog ready. Generate a session to inspect the planner.");
    }}

    async function loadHistory() {{
      const response = await fetch("/taste-os/history");
      if (!response.ok) {{
        throw new Error(`History request failed with status ${{response.status}}`);
      }}
      state.history = await response.json();
      renderHistory(state.history);
    }}

    async function submitFeedback(sessionId, artistName, signal) {{
      if (!sessionId) {{
        setStatus("Generate a session before recording feedback.", true);
        return;
      }}
      setStatus(`Recording ${{signal}} for ${{artistName}}...`);
      const response = await fetch("/taste-os/feedback", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
        }},
        body: JSON.stringify({{
          session_id: sessionId,
          artist_name: artistName,
          signal,
        }}),
      }});
      const data = await response.json();
      if (!response.ok) {{
        const message = data?.error?.message || `Feedback request failed with status ${{response.status}}`;
        setStatus(message, true);
        return;
      }}
      await loadHistory();
      if (state.payload) {{
        state.payload.memory_summary = data.feedback_memory || state.payload.memory_summary;
        renderSummary(state.payload);
        renderResults(state.payload);
      }}
      setStatus(`Recorded ${{signal}} for ${{artistName}}.`);
    }}

    async function runSession() {{
      runButton.disabled = true;
      setStatus("Generating session...");
      try {{
        const payload = {{
          mode: state.mode,
          scenario: state.scenario,
          top_k: Math.min(config.maxTopK, Math.max(1, Number(topKInput.value || config.defaultTopK))),
          include_video: includeVideoInput.checked,
          persist_artifacts: persistArtifactsInput.checked,
          use_feedback_memory: useFeedbackMemoryInput.checked,
        }};
        const recentArtists = recentArtistsInput.value.trim();
        if (recentArtists) {{
          payload.recent_artists = recentArtists;
        }}
        const response = await fetch("/taste-os/session", {{
          method: "POST",
          headers: {{
            "Content-Type": "application/json",
          }},
          body: JSON.stringify(payload),
        }});
        const data = await response.json();
        if (!response.ok) {{
          const message = data?.error?.message || `Session request failed with status ${{response.status}}`;
          throw new Error(message);
        }}
        state.payload = data;
        renderSummary(data);
        renderResults(data);
        await loadHistory();
        setStatus(`Session ready for ${{data.request.mode}} / ${{data.request.scenario}}.`);
      }} catch (error) {{
        setStatus(error.message || "Taste OS session failed.", true);
      }} finally {{
        runButton.disabled = false;
      }}
    }}

    runButton.addEventListener("click", runSession);
    refreshMemoryButton.addEventListener("click", () => {{
      loadHistory().then(() => {{
        setStatus("Memory refreshed.");
      }}).catch((error) => {{
        setStatus(error.message || "Memory refresh failed.", true);
      }});
    }});
    Promise.all([loadCatalog(), loadHistory()]).catch((error) => {{
      setStatus(error.message || "Taste OS studio failed to load.", true);
    }});
  </script>
</body>
</html>"""
