import html
import json
import pathlib
import logging

logger = logging.getLogger(__name__)


def generate_html_report(mismatch_report_file, output_file=None):
    report_path = pathlib.Path(mismatch_report_file)
    output_path = pathlib.Path(output_file) if output_file else report_path.with_suffix(".html")

    with open(report_path, encoding="utf-8") as json_file:
        rows = json.load(json_file)

    counts = {}
    for row in rows:
        status = row.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1

    summary_items = "\n".join(
        f"<span class=\"summary-item {html.escape(status.lower())}\">"
        f"{html.escape(status)}: {count}</span>"
        for status, count in sorted(counts.items())
    )

    table_rows = []
    for row in rows:
        score = row.get("score")
        score_text = "" if score is None else f"{score:.3f}"
        status = row.get("status", "UNKNOWN")
        table_rows.append(
            "<tr>"
            f"<td>{row.get('index', '')}</td>"
            f"<td>{row.get('start', '')}</td>"
            f"<td>{row.get('end', '')}</td>"
            f"<td>{row.get('timestamp', '')}</td>"
            f"<td>{html.escape(str(row.get('audio_text', '')))}</td>"
            f"<td>{html.escape(str(row.get('ocr_text', '')))}</td>"
            f"<td>{score_text}</td>"
            f"<td><span class=\"status {html.escape(status.lower())}\">{html.escape(status)}</span></td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Subtitle Mismatch Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            color: #202124;
            background: #f8f9fa;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 28px;
        }}
        .meta {{
            margin-bottom: 20px;
            color: #5f6368;
        }}
        .summary {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 24px;
        }}
        .summary-item,
        .status {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 12px;
        }}
        .ok {{
            background: #dff4e5;
            color: #137333;
        }}
        .review {{
            background: #fef7d1;
            color: #8a5a00;
        }}
        .mismatch,
        .ocr_failed {{
            background: #fce8e6;
            color: #a50e0e;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #ffffff;
        }}
        th,
        td {{
            border: 1px solid #dadce0;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background: #eef2f7;
            position: sticky;
            top: 0;
        }}
        td:nth-child(5),
        td:nth-child(6) {{
            min-width: 260px;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <h1>Subtitle Mismatch Report</h1>
    <div class="meta">{html.escape(str(report_path))}</div>
    <div class="summary">{summary_items}</div>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Start</th>
                <th>End</th>
                <th>OCR Timestamp</th>
                <th>Audio Transcript</th>
                <th>OCR Text</th>
                <th>Score</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as html_file:
        html_file.write(document)

    return str(output_path)
