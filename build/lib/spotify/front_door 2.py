from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from .branch_portfolio import build_branch_portfolio_report, write_branch_portfolio_artifacts
from .claim_to_demo import build_claim_to_demo_report, write_claim_to_demo_artifacts
from .run_artifacts import copy_file_if_changed, write_json, write_markdown


def _coerce_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _file_href(raw_path: object) -> str:
    path_text = str(raw_path).strip()
    if not path_text:
        return "#"
    path = Path(path_text).expanduser()
    if not path.exists():
        return "#"
    return path.resolve().as_uri()


def build_front_door_report(output_dir: Path | str = "outputs") -> dict[str, object]:
    output_root = Path(output_dir).expanduser().resolve()
    claim_report = build_claim_to_demo_report(output_root)
    branch_report = build_branch_portfolio_report(output_root)
    hero = _coerce_dict(claim_report.get("flagship_demo"))
    primary_claim = _coerce_dict(claim_report.get("primary_claim"))

    branch_cards = []
    for branch in _coerce_list(branch_report.get("branches")):
        if not isinstance(branch, dict):
            continue
        branch_cards.append(
            {
                "label": str(branch.get("label", "")),
                "status": str(branch.get("status", "")),
                "audience": str(branch.get("audience", "")),
                "success_metric": str(branch.get("success_metric", "")),
                "entry_command": str(branch.get("entry_command", "")),
                "live_signal": str(branch.get("live_signal", "")),
                "artifacts": [str(item) for item in _coerce_list(branch.get("artifacts"))[:3]],
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "title": "Spotify Personal Taste OS",
        "subtitle": "One front door for the strongest product demo, operating review, creator strategy surface, and research claim.",
        "hero": {
            "headline": str(claim_report.get("headline", "")),
            "flagship_label": str(hero.get("label", "")),
            "flagship_mode": str(hero.get("mode", "")),
            "flagship_scenario": str(hero.get("scenario", "")),
            "flagship_story": str(hero.get("story", "")),
            "flagship_outcome": str(hero.get("story_outcome", "")),
            "top_artist": str(hero.get("top_artist", "")),
            "fallback_policy_name": str(hero.get("fallback_policy_name", "")),
            "adaptive_replans": int(hero.get("adaptive_replans", 0) or 0),
            "adaptive_safe_route_steps": int(hero.get("adaptive_safe_route_steps", 0) or 0),
            "primary_claim_key": str(primary_claim.get("key", "")),
            "primary_claim_title": str(primary_claim.get("title", "")),
            "primary_claim_status": str(primary_claim.get("status", "")),
            "primary_claim_summary": str(primary_claim.get("summary", "")),
        },
        "spotlight_metrics": [
            dict(item)
            for item in _coerce_list(claim_report.get("evidence_scoreboard"))
            if isinstance(item, dict)
        ][:6],
        "bridge_points": [str(item) for item in _coerce_list(claim_report.get("bridge_points"))],
        "review_sequence": [dict(item) for item in _coerce_list(claim_report.get("review_sequence")) if isinstance(item, dict)],
        "next_actions": [str(item) for item in _coerce_list(claim_report.get("next_actions"))[:4]],
        "branch_cards": branch_cards,
        "claim_to_demo": claim_report,
        "branch_report": branch_report,
    }


def write_front_door_artifacts(report: dict[str, object], *, output_dir: Path | str = "outputs") -> dict[str, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    artifact_root = output_root / "analysis" / "front_door"
    artifact_root.mkdir(parents=True, exist_ok=True)

    claim_paths = write_claim_to_demo_artifacts(_coerce_dict(report.get("claim_to_demo")), output_dir=output_root)
    branch_paths = write_branch_portfolio_artifacts(_coerce_dict(report.get("branch_report")), output_dir=output_root)

    copied_artifacts = {
        "claim_to_demo_md": str(copy_file_if_changed(claim_paths["md"], artifact_root / "flagship" / "claim_to_demo.md").resolve()),
        "claim_to_demo_talk_track_md": str(
            copy_file_if_changed(claim_paths["talk_track_md"], artifact_root / "flagship" / "claim_to_demo_talk_track.md").resolve()
        ),
        "branch_portfolio_md": str(copy_file_if_changed(branch_paths["md"], artifact_root / "portfolio" / "portfolio_branches.md").resolve()),
    }

    payload = {**report, "copied_artifacts": copied_artifacts}
    json_path = write_json(artifact_root / "front_door.json", payload)

    md_lines = [
        "# Front Door",
        "",
        f"- Generated at: `{report.get('generated_at', '')}`",
        f"- Title: {report.get('title', '')}",
        f"- Subtitle: {report.get('subtitle', '')}",
        "",
        "## Flagship",
        "",
    ]
    hero = _coerce_dict(report.get("hero"))
    md_lines.append(f"- Demo: `{hero.get('flagship_label', '')}`")
    md_lines.append(f"- Claim: `{hero.get('primary_claim_key', '')}` [{hero.get('primary_claim_status', '')}]")
    md_lines.append(f"- Story: {hero.get('flagship_story', '')}")
    md_lines.append(f"- Outcome: {hero.get('flagship_outcome', '')}")
    md_lines.extend(["", "## Open First", ""])
    md_lines.append(f"- Claim-to-demo pack: `{copied_artifacts['claim_to_demo_md']}`")
    md_lines.append(f"- Talk track: `{copied_artifacts['claim_to_demo_talk_track_md']}`")
    md_lines.append(f"- Portfolio branches: `{copied_artifacts['branch_portfolio_md']}`")
    md_path = write_markdown(artifact_root / "front_door.md", md_lines)

    metric_cards = []
    for row in _coerce_list(report.get("spotlight_metrics")):
        if not isinstance(row, dict):
            continue
        metric_cards.append(
            f"""
            <article class="metric-card">
              <p class="metric-label">{escape(str(row.get("label", "")))}</p>
              <p class="metric-value">{escape(str(row.get("formatted_value", "")))}</p>
              <p class="metric-why">{escape(str(row.get("why_it_matters", "")))}</p>
            </article>
            """.strip()
        )

    branch_cards = []
    for card in _coerce_list(report.get("branch_cards")):
        if not isinstance(card, dict):
            continue
        artifact_links = "".join(
            f'<li><a href="{escape(_file_href(path))}">{escape(Path(str(path)).name or str(path))}</a></li>'
            for path in _coerce_list(card.get("artifacts"))
            if str(path).strip()
        )
        branch_cards.append(
            f"""
            <article class="branch-card">
              <div class="branch-card-top">
                <span class="status-pill status-{escape(str(card.get("status", "")))}">{escape(str(card.get("status", "")))}</span>
                <p class="branch-command">{escape(str(card.get("entry_command", "")))}</p>
              </div>
              <h3>{escape(str(card.get("label", "")))}</h3>
              <p class="branch-audience">{escape(str(card.get("audience", "")))}</p>
              <p class="branch-signal">{escape(str(card.get("live_signal", "")))}</p>
              <p class="branch-success"><strong>Success metric:</strong> {escape(str(card.get("success_metric", "")))}</p>
              <ul class="branch-links">{artifact_links}</ul>
            </article>
            """.strip()
        )

    review_steps = []
    for row in _coerce_list(report.get("review_sequence")):
        if not isinstance(row, dict):
            continue
        review_steps.append(
            f"""
            <li class="review-step">
              <span class="review-index">{escape(str(row.get("step", "")))}</span>
              <div>
                <p class="review-label">{escape(str(row.get("label", "")))}</p>
                <p class="review-why">{escape(str(row.get("why", "")))}</p>
                <a class="review-link" href="{escape(_file_href(row.get("artifact", "")))}">Open artifact</a>
              </div>
            </li>
            """.strip()
        )

    bridge_points = "".join(f"<li>{escape(str(item))}</li>" for item in _coerce_list(report.get("bridge_points")))
    next_actions = "".join(f"<li>{escape(str(item))}</li>" for item in _coerce_list(report.get("next_actions")))
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(str(report.get("title", "")))}</title>
    <style>
      :root {{
        --paper: #f6f0e5;
        --ink: #11221d;
        --accent: #d96c3d;
        --accent-soft: #f4a261;
        --teal: #1f6f78;
        --card: rgba(255, 252, 247, 0.88);
        --border: rgba(17, 34, 29, 0.12);
        --shadow: 0 18px 40px rgba(17, 34, 29, 0.10);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(217, 108, 61, 0.18), transparent 34%),
          radial-gradient(circle at top right, rgba(31, 111, 120, 0.18), transparent 32%),
          linear-gradient(180deg, #fff9f1 0%, var(--paper) 100%);
      }}
      .page {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 40px 0 64px; }}
      .hero {{ display: grid; grid-template-columns: 1.35fr 0.95fr; gap: 24px; align-items: stretch; }}
      .hero-card, .panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); }}
      .hero-card {{ padding: 36px; position: relative; overflow: hidden; }}
      .hero-card::after {{ content: ""; position: absolute; right: -40px; top: -40px; width: 180px; height: 180px; border-radius: 999px; background: linear-gradient(135deg, rgba(217, 108, 61, 0.18), rgba(31, 111, 120, 0.18)); }}
      .eyebrow {{ display: inline-block; margin: 0 0 16px; padding: 8px 14px; border-radius: 999px; background: rgba(17, 34, 29, 0.06); letter-spacing: 0.08em; text-transform: uppercase; font-size: 12px; }}
      h1, h2, h3 {{ font-family: "Iowan Old Style", "Palatino Linotype", serif; margin: 0; }}
      h1 {{ font-size: clamp(2.5rem, 5vw, 4.5rem); line-height: 0.98; max-width: 10ch; }}
      .subtitle {{ max-width: 48rem; font-size: 1.08rem; line-height: 1.6; margin: 18px 0 28px; }}
      .hero-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
      .hero-stat {{ padding: 16px; border-radius: 18px; background: rgba(255, 255, 255, 0.7); border: 1px solid rgba(17, 34, 29, 0.08); }}
      .hero-stat-label {{ font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; color: rgba(17, 34, 29, 0.68); margin: 0 0 6px; }}
      .hero-stat-value {{ font-size: 1.15rem; font-weight: 700; margin: 0; }}
      .panel {{ padding: 28px; }}
      .panel h2 {{ font-size: 1.5rem; margin-bottom: 12px; }}
      .review-list, .bridge-list, .next-actions, .branch-links {{ margin: 0; padding-left: 18px; }}
      .review-step {{ display: grid; grid-template-columns: 36px 1fr; gap: 14px; margin-bottom: 16px; list-style: none; }}
      .review-index {{ width: 36px; height: 36px; display: grid; place-items: center; border-radius: 999px; background: linear-gradient(135deg, var(--accent), var(--accent-soft)); color: white; font-weight: 700; }}
      .review-label {{ font-weight: 700; margin: 0 0 4px; }}
      .review-why {{ margin: 0 0 6px; color: rgba(17, 34, 29, 0.76); }}
      .review-link, a {{ color: var(--teal); text-decoration: none; }}
      .review-link:hover, a:hover {{ text-decoration: underline; }}
      .section {{ margin-top: 28px; }}
      .section-head {{ display: flex; justify-content: space-between; align-items: end; gap: 16px; margin-bottom: 16px; }}
      .section-head p {{ margin: 0; color: rgba(17, 34, 29, 0.68); }}
      .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
      .metric-card {{ padding: 18px; border-radius: 20px; background: rgba(255, 255, 255, 0.72); border: 1px solid rgba(17, 34, 29, 0.08); }}
      .metric-label {{ margin: 0 0 8px; font-size: 0.9rem; color: rgba(17, 34, 29, 0.72); }}
      .metric-value {{ margin: 0 0 8px; font-size: 2rem; font-weight: 700; }}
      .metric-why {{ margin: 0; line-height: 1.5; color: rgba(17, 34, 29, 0.8); }}
      .branch-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
      .branch-card {{ padding: 22px; border-radius: 22px; background: rgba(255, 255, 255, 0.76); border: 1px solid rgba(17, 34, 29, 0.08); }}
      .branch-card-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 10px; }}
      .status-pill {{ display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 999px; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }}
      .status-ready, .status-ready_with_gaps {{ background: rgba(31, 111, 120, 0.12); color: var(--teal); }}
      .status-attention {{ background: rgba(217, 108, 61, 0.14); color: var(--accent); }}
      .status-missing {{ background: rgba(17, 34, 29, 0.08); color: rgba(17, 34, 29, 0.72); }}
      .branch-command {{ margin: 0; font-size: 0.8rem; color: rgba(17, 34, 29, 0.62); }}
      .branch-card h3 {{ font-size: 1.45rem; margin-bottom: 8px; }}
      .branch-audience, .branch-signal, .branch-success {{ margin: 0 0 10px; line-height: 1.55; }}
      @media (max-width: 900px) {{
        .hero, .metrics, .branch-grid {{ grid-template-columns: 1fr; }}
        .hero-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <article class="hero-card">
          <p class="eyebrow">Front Door</p>
          <h1>{escape(str(report.get("title", "")))}</h1>
          <p class="subtitle">{escape(str(report.get("subtitle", "")))}</p>
          <div class="hero-grid">
            <div class="hero-stat"><p class="hero-stat-label">Flagship Demo</p><p class="hero-stat-value">{escape(str(hero.get("flagship_label", "")))}</p></div>
            <div class="hero-stat"><p class="hero-stat-label">Primary Claim</p><p class="hero-stat-value">{escape(str(hero.get("primary_claim_key", "")))}</p></div>
            <div class="hero-stat"><p class="hero-stat-label">Opening Artist</p><p class="hero-stat-value">{escape(str(hero.get("top_artist", "")))}</p></div>
          </div>
        </article>
        <aside class="panel">
          <h2>Review This In Three Moves</h2>
          <ol class="review-list">{''.join(review_steps)}</ol>
        </aside>
      </section>

      <section class="section">
        <div class="section-head"><div><h2>Why The Flagship Works</h2><p>{escape(str(hero.get("flagship_story", "")))}</p></div></div>
        <article class="panel">
          <p><strong>Outcome:</strong> {escape(str(hero.get("flagship_outcome", "")))}</p>
          <p><strong>Policy route:</strong> {escape(str(hero.get("fallback_policy_name", "")))}</p>
          <p><strong>Research bridge:</strong> {escape(str(hero.get("primary_claim_title", "")))}</p>
          <ul class="bridge-list">{bridge_points}</ul>
        </article>
      </section>

      <section class="section">
        <div class="section-head"><div><h2>Evidence At A Glance</h2><p>The fastest numbers to scan before opening the deeper packs.</p></div></div>
        <div class="metrics">{''.join(metric_cards)}</div>
      </section>

      <section class="section">
        <div class="section-head"><div><h2>Four Branches</h2><p>The repo is strongest when every new feature clearly strengthens one of these lanes.</p></div></div>
        <div class="branch-grid">{''.join(branch_cards)}</div>
      </section>

      <section class="section">
        <div class="section-head"><div><h2>Next Moves</h2><p>These are the current follow-through items after the landing pass.</p></div></div>
        <article class="panel">
          <ul class="next-actions">{next_actions}</ul>
          <p><a href="{escape(_file_href(copied_artifacts["claim_to_demo_md"]))}">Open the full claim-to-demo pack</a> or <a href="{escape(_file_href(copied_artifacts["branch_portfolio_md"]))}">open the branch portfolio</a>.</p>
        </article>
      </section>
    </main>
  </body>
</html>
"""
    html_path = artifact_root / "index.html"
    html_path.write_text(html, encoding="utf-8")

    return {
        "json": json_path,
        "md": md_path,
        "html": html_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.front_door",
        description="Build a front-door landing page for the strongest Taste OS, control-room, creator, and research artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output root containing analysis/, analytics/, history/, and runs/ subdirectories.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    report = build_front_door_report(output_root)
    paths = write_front_door_artifacts(report, output_dir=output_root)
    print(f"front_door_json={paths['json']}")
    print(f"front_door_md={paths['md']}")
    print(f"front_door_html={paths['html']}")
    print(f"flagship_demo={_coerce_dict(report.get('hero')).get('flagship_label', '')}")
    print(f"primary_claim={_coerce_dict(report.get('hero')).get('primary_claim_key', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
