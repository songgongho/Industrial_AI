"""Generate static HTML dashboard from predictions.

Usage:
    python scripts/generate_html_report.py --predictions outputs/predictions.json --output app/dashboard.html

Creates a standalone HTML dashboard with visualizations (no server required).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def generate_html(
    predictions: list[dict[str, Any]],
    title: str = "Press Defect Prediction Dashboard",
) -> str:
    """Generate HTML dashboard from predictions.

    Args:
        predictions: List of prediction dictionaries
        title: Dashboard title

    Returns:
        HTML string
    """
    # Calculate statistics
    total = len(predictions)
    defects = sum(1 for p in predictions if p.get("prediction") == "DEFECTIVE")
    normal = total - defects
    defect_rate = (defects / total * 100) if total > 0 else 0

    # Extract probabilities for histogram
    probs = [p.get("defect_probability", 0) for p in predictions if p.get("defect_probability") is not None]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 14px;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .metric {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .chart {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .table-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #f5f5f5;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        tr:hover {{
            background: #f9f9f9;
        }}
        
        .status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .defective {{
            background: #ffebee;
            color: #c62828;
        }}
        
        .normal {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        
        footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            padding: 20px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Real-time Press Defect Prediction Results</p>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{total:,}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Defects Detected</div>
                    <div class="metric-value">{defects}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Defect Rate</div>
                    <div class="metric-value">{defect_rate:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Normal Samples</div>
                    <div class="metric-value">{normal:,}</div>
                </div>
            </div>
        </header>
        
        <div class="charts">
            <div class="chart">
                <div id="pie-chart"></div>
            </div>
            <div class="chart">
                <div id="histogram-chart"></div>
            </div>
        </div>
        
        <div class="table-container">
            <h2 style="margin-bottom: 20px;">Detailed Predictions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Sample ID</th>
                        <th>Defect Probability</th>
                        <th>Status</th>
                        <th>Anomaly Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add sample rows (max 100 for performance)
    for pred in predictions[:100]:
        sample_id = pred.get("sample_id", "N/A")
        prob = pred.get("defect_probability", 0) or 0
        status = pred.get("prediction", "UNKNOWN")
        anomaly_conf = pred.get("anomaly_confidence", 0) or 0
        status_class = "defective" if status == "DEFECTIVE" else "normal"

        html += f"""                    <tr>
                        <td>#{sample_id}</td>
                        <td>{prob:.4f}</td>
                        <td><span class="status {status_class}">{status}</span></td>
                        <td>{anomaly_conf:.4f}</td>
                    </tr>
"""

    if total > 100:
        html += f"""                    <tr>
                        <td colspan="4" style="text-align: center; color: #999;">
                            ... and {total - 100} more rows
                        </td>
                    </tr>
"""

    html += """                </tbody>
            </table>
        </div>
        
        <footer>
            <p>Generated by MS-CDPNet Press Defect Prediction System</p>
            <p>For more information, visit: <a href="https://github.com/your-username/pcb-lamination-press-defect-prediction" style="color: white;">GitHub Repository</a></p>
        </footer>
    </div>
    
    <script>
        // Pie chart
        var pieData = [{
            values: [""" + str(defects) + """, """ + str(normal) + """],
            labels: ['Defects', 'Normal'],
            type: 'pie',
            marker: {
                colors: ['#f44336', '#4caf50']
            }
        }];
        
        var pieLayout = {
            title: 'Defect Distribution',
            height: 400
        };
        
        Plotly.newPlot('pie-chart', pieData, pieLayout, {responsive: true});
        
        // Histogram
        var histogramData = [{
            x: """ + json.dumps(probs) + """,
            type: 'histogram',
            nbinsx: 30,
            marker: {
                color: '#667eea'
            }
        }];
        
        var histogramLayout = {
            title: 'Defect Probability Distribution',
            xaxis: {title: 'Probability'},
            yaxis: {title: 'Count'},
            height: 400
        };
        
        Plotly.newPlot('histogram-chart', histogramData, histogramLayout, {responsive: true});
    </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/predictions.json",
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="app/dashboard.html",
        help="Output HTML file path (default: app/dashboard.html)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Press Defect Prediction Dashboard",
        help="Dashboard title",
    )

    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_path = Path(args.output)

    if not pred_path.exists():
        print(f"Error: Predictions file not found: {pred_path}")
        sys.exit(1)

    print(f"Reading predictions from {pred_path}...")
    with open(pred_path) as f:
        predictions = json.load(f)

    print(f"Generating HTML report with {len(predictions)} predictions...")
    html = generate_html(predictions, title=args.title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    print(f"✓ HTML report saved to {output_path}")
    print(f"  Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()

