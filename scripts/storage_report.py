from __future__ import annotations

import argparse
from pathlib import Path

from spotify.storage_report import build_storage_report, write_storage_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a storage report for Spotify project outputs.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Project output directory.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top runs/files to include.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    report = build_storage_report(output_dir, top_n=max(1, int(args.top_n)))
    json_path, md_path = write_storage_report(output_dir, top_n=max(1, int(args.top_n)))

    print(f"Storage report written to {json_path}")
    print(f"Markdown summary written to {md_path}")
    print(f"Total outputs size: {report['human_size']}")
    print("Top sections:")
    for row in report["section_totals"][:5]:
        print(f"- {row['section']}: {row['human_size']}")
    print("Top categories:")
    for row in report["category_totals"][:5]:
        print(f"- {row['category']}: {row['human_size']} ({row['file_count']} files)")
    print("Largest runs:")
    for row in report["runs"][:5]:
        print(f"- {row['run_id']}: {row['human_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
