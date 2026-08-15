from __future__ import annotations

from pathlib import Path


def save_missing_values_plot(missing_report: list[dict], output_path: str | Path, top_n: int = 30) -> None:
    """Save a missing-values bar plot from Spark-generated report rows."""
    import matplotlib.pyplot as plt

    rows = sorted(missing_report, key=lambda r: float(r.get("missing_ratio", 0.0)), reverse=True)[:top_n]
    columns = [str(r["column"]) for r in rows]
    ratios = [float(r["missing_ratio"]) for r in rows]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(columns))))
    ax.barh(columns, ratios)
    ax.set_xlabel("Missing ratio")
    ax.set_ylabel("Column")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
