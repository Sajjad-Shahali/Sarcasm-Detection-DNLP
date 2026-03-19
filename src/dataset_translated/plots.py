# plot_results.py
# Plots results from the NEW evaluation outputs:
#   - {prefix}_per_variety.csv
#   - {prefix}_per_source.csv
#   - {prefix}_per_variety_source.csv

from pathlib import Path
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=r".\src\dataset_translated\results")
    parser.add_argument("--output_prefix", type=str, default="tdata_")
    parser.add_argument("--plots_dir", type=str, default=r".\plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Filenames exactly matching evaluation.py
    variety_path = results_dir / f"{args.output_prefix}_per_variety.csv"
    source_path = results_dir / f"{args.output_prefix}_per_source.csv"
    vs_path = results_dir / f"{args.output_prefix}_per_variety_source.csv"

    if not variety_path.exists():
        raise FileNotFoundError(f"Missing file: {variety_path}")

    sns.set(style="whitegrid", font_scale=1.1)

    # -----------------------------
    # Load per-variety
    # -----------------------------
    df_var = pd.read_csv(variety_path)

    # Filter real varieties (exclude meta rows like __ROBUSTNESS_GAP__)
    plot_var = df_var[~df_var["variety"].astype(str).str.startswith("__")].copy()

    # ======================================================
    # 1) Macro-F1 per Variety
    # ======================================================
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_var, x="variety", y="f1_macro", hue="model")
    plt.title("Macro-F1 per English Variety")
    plt.ylabel("Macro-F1")
    plt.xlabel("Variety")
    plt.xticks(rotation=30)
    plt.legend(title="Model")
    plt.tight_layout()
    out1 = plots_dir / f"{args.output_prefix}_chart_f1_per_variety.png"
    plt.savefig(out1, dpi=300)
    plt.show()

    # ======================================================
    # 2) Robustness Gap (max–min F1 across varieties)
    # ======================================================
    gap_df = df_var[df_var["variety"] == "__ROBUSTNESS_GAP__"].copy()
    if len(gap_df) > 0:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=gap_df, x="model", y="f1_macro")
        plt.title("Cross-Variety Robustness Gap")
        plt.ylabel("Max–Min Macro-F1")
        plt.xlabel("Model")
        plt.tight_layout()
        out2 = plots_dir / f"{args.output_prefix}_chart_robustness_gap.png"
        plt.savefig(out2, dpi=300)
        plt.show()
    else:
        print("No __ROBUSTNESS_GAP__ row found in per_variety file -> skipping robustness plot.")

    # ======================================================
    # 3) Precision vs Recall (per Variety)
    # ======================================================
    if {"precision", "recall"}.issubset(plot_var.columns):
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=plot_var,
            x="precision",
            y="recall",
            hue="model",
            style="variety",
            s=120
        )
        plt.title("Precision vs Recall across Varieties")
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        out3 = plots_dir / f"{args.output_prefix}_chart_precision_recall_variety.png"
        plt.savefig(out3, dpi=300)
        plt.show()
    else:
        print("precision/recall not found -> skipping precision-recall (variety) plot.")

    # ======================================================
    # 4) Inference Latency per Model
    # (stored in f1_macro column by convention)
    # ======================================================
    lat_df = df_var[df_var["variety"] == "__LATENCY_MS_PER_SAMPLE__"].copy()
    if len(lat_df) > 0:
        plt.figure(figsize=(6, 4))
        sns.barplot(data=lat_df, x="model", y="f1_macro")
        plt.title("Inference Latency per Sample")
        plt.ylabel("Milliseconds")
        plt.xlabel("Model")
        plt.tight_layout()
        out4 = plots_dir / f"{args.output_prefix}_chart_latency.png"
        plt.savefig(out4, dpi=300)
        plt.show()
    else:
        print("No __LATENCY_MS_PER_SAMPLE__ row found -> skipping latency plot.")

    # ======================================================
    # 5) Summary table (mean ± std F1 across varieties)
    # ======================================================
    summary_var = (
        plot_var.groupby("model")
        .agg(mean_f1=("f1_macro", "mean"), std_f1=("f1_macro", "std"))
        .reset_index()
    )
    print("\n=== Summary (Variety): Mean ± Std Macro-F1 per Model ===")
    print(summary_var)
    summary_var.to_csv(plots_dir / f"{args.output_prefix}_summary_mean_std_f1_variety.csv", index=False)

    # -----------------------------
    # Load per-source (optional)
    # -----------------------------
    if source_path.exists():
        df_src = pd.read_csv(source_path)

        # ======================================================
        # 6) Macro-F1 per Source
        # ======================================================
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_src, x="source", y="f1_macro", hue="model")
        plt.title("Macro-F1 per Source")
        plt.ylabel("Macro-F1")
        plt.xlabel("Source")
        plt.xticks(rotation=20)
        plt.legend(title="Model")
        plt.tight_layout()
        out5 = plots_dir / f"{args.output_prefix}_chart_f1_per_source.png"
        plt.savefig(out5, dpi=300)
        plt.show()

        # ======================================================
        # 7) Precision vs Recall (per Source)
        # ======================================================
        if {"precision", "recall"}.issubset(df_src.columns):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=df_src,
                x="precision",
                y="recall",
                hue="model",
                style="source",
                s=120
            )
            plt.title("Precision vs Recall across Sources")
            plt.xlabel("Precision")
            plt.ylabel("Recall")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            out6 = plots_dir / f"{args.output_prefix}_chart_precision_recall_source.png"
            plt.savefig(out6, dpi=300)
            plt.show()

        # Summary per source
        summary_src = (
            df_src.groupby("model")
            .agg(mean_f1=("f1_macro", "mean"), std_f1=("f1_macro", "std"))
            .reset_index()
        )
        print("\n=== Summary (Source): Mean ± Std Macro-F1 per Model ===")
        print(summary_src)
        summary_src.to_csv(plots_dir / f"{args.output_prefix}_summary_mean_std_f1_source.csv", index=False)
    else:
        print(f"No per-source file found at {source_path} -> skipping source plots.")

    # -----------------------------
    # Variety × Source heatmaps (optional)
    # -----------------------------
    if vs_path.exists():
        df_vs = pd.read_csv(vs_path)

        # One heatmap PER model (clean + readable)
        for model_name, g in df_vs.groupby("model"):
            pivot = g.pivot(index="variety", columns="source", values="f1_macro")

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Macro-F1"})
            plt.title(f"Macro-F1 Heatmap (Variety × Source) — {model_name}")
            plt.xlabel("Source")
            plt.ylabel("Variety")
            plt.tight_layout()

            out7 = plots_dir / f"{args.output_prefix}_heatmap_variety_source_{model_name}.png"
            plt.savefig(out7, dpi=300)
            plt.show()
    else:
        print(f"No variety×source file found at {vs_path} -> skipping heatmaps.")

    print(f"\nAll plots saved in: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
