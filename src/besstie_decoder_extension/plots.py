import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load results
# -----------------------------
df = pd.read_csv(r"src\besstie_decoder_extension\per_variety_results.csv")

# Filter real varieties (exclude meta rows)
plot_df = df[~df["variety"].str.startswith("__")].copy()

sns.set(style="whitegrid", font_scale=1.1)

# ======================================================
# 1) Macro-F1 per Variety (Main comparison chart)
# ======================================================
plt.figure(figsize=(11, 6))
sns.barplot(
    data=plot_df,
    x="variety",
    y="f1_macro",
    hue="model"
)
plt.title("Macro-F1 per English Variety")
plt.ylabel("Macro-F1")
plt.xlabel("Variety")
plt.xticks(rotation=30)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig(r"src\besstie_decoder_extension\plots\chart_f1_per_variety.png", dpi=300)
plt.show()

# ======================================================
# 2) Robustness Gap (max–min F1 across varieties)
# ======================================================
gap_df = df[df["variety"] == "__ROBUSTNESS_GAP__"]

plt.figure(figsize=(6, 4))
sns.barplot(
    data=gap_df,
    x="model",
    y="f1_macro"
)
plt.title("Cross-Variety Robustness Gap")
plt.ylabel("Max–Min Macro-F1")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(r"src\besstie_decoder_extension\plots\chart_robustness_gap.png", dpi=300)
plt.show()

# ======================================================
# 3) Precision vs Recall (Error behavior)
# ======================================================
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=plot_df,
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
plt.savefig(r"src\besstie_decoder_extension\plots\chart_precision_recall.png", dpi=300)
plt.show()

# ======================================================
# 4) Inference Latency per Model
# ======================================================
lat_df = df[df["variety"] == "__LATENCY_MS_PER_SAMPLE__"]

plt.figure(figsize=(6, 4))
sns.barplot(
    data=lat_df,
    x="model",
    y="f1_macro"
)
plt.title("Inference Latency per Sample")
plt.ylabel("Milliseconds")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig(r"src\besstie_decoder_extension\plots\chart_latency.png", dpi=300)
plt.show()

# ======================================================
# 5) Summary Table (mean ± std Macro-F1)
# ======================================================
summary = (
    plot_df
    .groupby("model")
    .agg(
        mean_f1=("f1_macro", "mean"),
        std_f1=("f1_macro", "std")
    )
    .reset_index()
)

print("\n=== Summary: Mean ± Std Macro-F1 per Model ===")
print(summary)

summary.to_csv(r"src\besstie_decoder_extension\summary_mean_std_f1.csv", index=False)
