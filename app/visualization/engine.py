import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
from app.utils.logging import get_logger

logger = get_logger("visualization.engine")

_PALETTE = sns.color_palette("muted")


def _style() -> None:
    avail = plt.style.available
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in avail else "ggplot"
    plt.style.use(style)
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "font.family": "sans-serif",
        "grid.alpha": 0.3,
    })


class VisualizationEngine:

    @staticmethod
    def generate_chart(
        df: pd.DataFrame,
        chart_type: str,
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        output_path: str = "chart.png",
        **_kwargs,
    ) -> str:
        _style()
        plt.close("all")
        ct = chart_type.lower().replace(" ", "_").replace("-", "_")

        if ct == "pair_plot":
            plt.close("all")
            num_cols = list(df.select_dtypes(include=[np.number]).columns[:5])
            cols = num_cols + ([hue] if hue and hue in df.columns else [])
            g = sns.pairplot(df[cols], hue=hue, palette=_PALETTE if hue else None)
            g.figure.suptitle(title or "Pair Plot", y=1.02, fontsize=14, fontweight="bold")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            g.figure.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            return os.path.abspath(output_path)

        fig, ax = plt.subplots()
        try:
            if ct == "histogram":
                if not x:
                    raise ValueError("x required for histogram")
                sns.histplot(data=df, x=x, hue=hue, kde=True, ax=ax,
                             palette=_PALETTE if hue else None)
                ax.set_title(title or f"Histogram — {x}")

            elif ct in ("bar_chart", "bar"):
                if not x:
                    raise ValueError("x required for bar chart")
                if y:
                    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=_PALETTE)
                    ax.set_title(title or f"Bar Chart — {y} vs {x}")
                else:
                    sns.countplot(data=df, x=x, hue=hue, ax=ax, palette=_PALETTE)
                    ax.set_title(title or f"Count Plot — {x}")
                plt.xticks(rotation=45, ha="right")

            elif ct in ("scatter_plot", "scatter"):
                if not x or not y:
                    raise ValueError("x and y required for scatter")
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax,
                                palette=_PALETTE if hue else None)
                ax.set_title(title or f"Scatter — {y} vs {x}")

            elif ct in ("line_chart", "line"):
                if not x or not y:
                    raise ValueError("x and y required for line chart")
                sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax,
                             palette=_PALETTE if hue else None)
                ax.set_title(title or f"Line — {y} vs {x}")

            elif ct in ("box_plot", "box"):
                if not x:
                    raise ValueError("x required for box plot")
                sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=_PALETTE)
                ax.set_title(title or f"Box Plot — {x}")

            elif ct in ("pie_chart", "pie"):
                if not x:
                    raise ValueError("x required for pie chart")
                grp = df.groupby(x)[y].sum().head(10) if y else df[x].value_counts().head(10)
                ax.pie(grp.values, labels=[str(i) for i in grp.index],
                       autopct="%1.1f%%", colors=_PALETTE)
                ax.axis("equal")
                ax.set_title(title or f"Pie — {x}")

            elif ct in ("heatmap", "correlation_matrix"):
                num = df.select_dtypes(include=[np.number])
                if num.empty:
                    raise ValueError("No numeric columns for heatmap")
                sns.heatmap(num.corr().fillna(0), annot=True, fmt=".2f",
                            cmap="coolwarm", ax=ax, square=True)
                ax.set_title(title or "Correlation Matrix")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)

            elif ct in ("violin_plot", "violin"):
                if not x:
                    raise ValueError("x required for violin plot")
                sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=_PALETTE)
                ax.set_title(title or f"Violin — {x}")

            elif ct in ("count_plot", "count"):
                if not x:
                    raise ValueError("x required for count plot")
                sns.countplot(data=df, x=x, hue=hue, ax=ax, palette=_PALETTE)
                ax.set_title(title or f"Count — {x}")
                plt.xticks(rotation=45, ha="right")

            elif ct in ("distribution_plot", "distribution", "dist"):
                if not x:
                    raise ValueError("x required for distribution plot")
                sns.kdeplot(data=df, x=x, hue=hue, fill=True, ax=ax,
                            palette=_PALETTE if hue else None)
                ax.set_title(title or f"Distribution — {x}")

            else:
                raise ValueError(f"Unsupported chart type: {chart_type!r}")

            plt.tight_layout()
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Chart saved → {output_path}")
            return os.path.abspath(output_path)

        finally:
            plt.close("all")
