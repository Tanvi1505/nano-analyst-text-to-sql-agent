"""
Generate Interactive Evaluation Dashboard
==========================================
Creates a beautiful HTML dashboard with charts showing:
- Self-correction impact analysis
- Error taxonomy
- Difficulty stratification
- Performance metrics

This is the portfolio-ready visualization for recruiters!
"""

import json
from pathlib import Path
from typing import Dict, List


def generate_html_dashboard(metrics: Dict, results: List[Dict], output_path: Path):
    """Generate interactive HTML dashboard with Chart.js."""

    # Extract data for charts
    difficulty_labels = []
    difficulty_accuracy = []
    difficulty_counts = []

    for diff in ["EASY", "MEDIUM", "HARD", "EXTRA_HARD"]:
        if diff in metrics.get("by_difficulty", {}):
            stats = metrics["by_difficulty"][diff]
            difficulty_labels.append(diff)
            difficulty_accuracy.append(round(stats["accuracy"] * 100, 1))
            difficulty_counts.append(stats["total"])

    # Self-correction data (for when model is loaded)
    attempt_labels = ["Attempt 1", "Attempt 2", "Attempt 3"]
    attempt_rates = [
        round(metrics.get("attempt_1_success_rate", 0) * 100, 1),
        round(metrics.get("attempt_2_success_rate", 0) * 100, 1),
        round(metrics.get("attempt_3_success_rate", 0) * 100, 1)
    ]

    # Error categories (if any)
    error_labels = []
    error_counts = []

    for error, count in metrics.get("top_errors", []):
        error_labels.append(error)
        error_counts.append(count)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nano-Analyst: Evaluation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}

        .header h1 {{
            font-size: 3em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .metric-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
        }}

        .metric-card h3 {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}

        .metric-card .value {{
            font-size: 3em;
            font-weight: 700;
            color: #667eea;
        }}

        .metric-card .label {{
            font-size: 0.9em;
            color: #999;
            margin-top: 5px;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}

        .chart-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .chart-card h2 {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
        }}

        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.8;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-top: 10px;
        }}

        .badge-success {{
            background: #10b981;
            color: white;
        }}

        .badge-warning {{
            background: #f59e0b;
            color: white;
        }}

        .details-table {{
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }}

        .details-table th,
        .details-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        .details-table th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}

        .details-table tr:hover {{
            background: #f9fafb;
        }}

        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 Nano-Analyst</h1>
            <p>SQL Agent Evaluation Dashboard</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Private, On-Device Text-to-SQL with Agentic Self-Correction
            </p>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Execution Accuracy</h3>
                <div class="value">{round(metrics.get('execution_accuracy', 0) * 100, 1)}%</div>
                <div class="label">{metrics.get('total_examples', 0)} examples</div>
                <span class="badge badge-success">EX Metric</span>
            </div>

            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="value">{round(metrics.get('success_rate', 0) * 100, 1)}%</div>
                <div class="label">Queries executed</div>
            </div>

            <div class="metric-card">
                <h3>Self-Correction Gain</h3>
                <div class="value">{round(metrics.get('self_correction_gain', 0) * 100, 1)}%</div>
                <div class="label">Recovered from errors</div>
                <span class="badge badge-warning">Agentic</span>
            </div>

            <div class="metric-card">
                <h3>Avg Execution Time</h3>
                <div class="value">{round(metrics.get('avg_execution_time_ms', metrics.get('avg_execution_time', 0)), 2)}</div>
                <div class="label">milliseconds</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-grid">
            <!-- Difficulty Breakdown -->
            <div class="chart-card">
                <h2>📊 Performance by Query Difficulty</h2>
                <div class="chart-container">
                    <canvas id="difficultyChart"></canvas>
                </div>
            </div>

            <!-- Self-Correction Analysis -->
            <div class="chart-card">
                <h2>🔄 Self-Correction Impact</h2>
                <div class="chart-container">
                    <canvas id="selfCorrectionChart"></canvas>
                </div>
                <p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 15px;">
                    Success rate by attempt number (shows error recovery)
                </p>
            </div>
        </div>

        <!-- Difficulty Details Table -->
        <div class="chart-card">
            <h2>📈 Detailed Performance Breakdown</h2>
            <table class="details-table">
                <thead>
                    <tr>
                        <th>Difficulty Level</th>
                        <th>Total Queries</th>
                        <th>Correct</th>
                        <th>Accuracy</th>
                        <th>Avg Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add table rows
    for diff in ["EASY", "MEDIUM", "HARD", "EXTRA_HARD"]:
        if diff in metrics.get("by_difficulty", {}):
            stats = metrics["by_difficulty"][diff]
            accuracy_pct = round(stats["accuracy"] * 100, 1)

            # Get avg time for this difficulty
            diff_results = [r for r in results if r.get("difficulty") == diff]
            avg_time = sum(r.get("execution_time", 0) for r in diff_results) / len(diff_results) if diff_results else 0

            html_content += f"""
                    <tr>
                        <td><strong>{diff}</strong></td>
                        <td>{stats['total']}</td>
                        <td>{stats['correct']}</td>
                        <td>{accuracy_pct}%</td>
                        <td>{round(avg_time * 1000, 2)}</td>
                    </tr>
            """

    html_content += f"""
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Built with Llama-3-8B (Fine-Tuned) + RAG + Agentic Self-Correction</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Framework: Unsloth QLoRA • Vector DB: ChromaDB • Dataset: Spider
            </p>
        </div>
    </div>

    <!-- Chart.js Scripts -->
    <script>
        // Difficulty Chart
        const difficultyCtx = document.getElementById('difficultyChart').getContext('2d');
        new Chart(difficultyCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(difficulty_labels)},
                datasets: [{{
                    label: 'Accuracy (%)',
                    data: {json.dumps(difficulty_accuracy)},
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',  // Green
                        'rgba(59, 130, 246, 0.8)',  // Blue
                        'rgba(245, 158, 11, 0.8)',  // Orange
                        'rgba(239, 68, 68, 0.8)'    // Red
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(59, 130, 246, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});

        // Self-Correction Chart
        const selfCorrectionCtx = document.getElementById('selfCorrectionChart').getContext('2d');
        new Chart(selfCorrectionCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(attempt_labels)},
                datasets: [{{
                    label: 'Success Rate (%)',
                    data: {json.dumps(attempt_rates)},
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 6,
                    pointBackgroundColor: 'rgba(102, 126, 234, 1)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✨ Interactive dashboard generated: {output_path}")


def main():
    """Generate dashboard from evaluation results."""

    project_root = Path.home() / "nano-analyst"
    eval_dir = project_root / "evaluation_results"

    # Load metrics and results
    metrics_path = eval_dir / "demo_metrics.json"
    results_path = eval_dir / "demo_results.json"

    if not metrics_path.exists():
        print("⚠️  No evaluation results found. Run demo_evaluation.py first!")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Generate dashboard
    dashboard_path = eval_dir / "dashboard.html"
    generate_html_dashboard(metrics, results, dashboard_path)

    print(f"\n✅ Dashboard ready!")
    print(f"\nOpen in browser:")
    print(f"  open {dashboard_path}")
    print(f"\nOr double-click the file in Finder")


if __name__ == "__main__":
    main()
