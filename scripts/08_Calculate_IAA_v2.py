import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Inter-annotator agreement for the 120-item rebuttal sample (2 annotators, all items labeled
# by both -- see Human_Annotation_Sample_v2_Annotator{1,2}.xlsx).
#
# Metric choice:
#   - Intensity (1-5) and Valence (1-5) are ORDINAL -> quadratic-weighted Cohen's Kappa +
#     Krippendorff's alpha with an ordinal distance metric. Unweighted Kappa would treat a
#     3-vs-4 disagreement the same as a 1-vs-5 disagreement, which is wrong for a Likert scale.
#   - Temporality is NOMINAL (no inherent order between the 3 categories) -> unweighted
#     Cohen's Kappa + Krippendorff's alpha with a nominal metric.
#
# Krippendorff's alpha is reported as the primary figure (more robust to skewed label
# distributions and small samples than Kappa; see Artstein & Poesio 2008), with weighted/plain
# Kappa alongside since it's the more familiar number to most reviewers.

ANNOTATOR1_FILE = "../data/human_annotation/Human_Annotation_Sample_v2_Annotator1.xlsx"
ANNOTATOR2_FILE = "../data/human_annotation/Human_Annotation_Sample_v2_Annotator2.xlsx"

ORDINAL_COLS = {
    "Human_Intensity (1-5)": "ordinal",
    "Human_Valence (1-5)": "ordinal",
}
NOMINAL_COLS = {
    "Human_Temporality": "nominal",
}


def load_annotator(path, suffix):
    df = pd.read_excel(path)
    keep = ["Sample_ID"] + list(ORDINAL_COLS) + list(NOMINAL_COLS)
    df = df[keep].rename(columns={c: f"{c}__{suffix}" for c in keep if c != "Sample_ID"})
    return df


def encode_for_krippendorff(a1, a2):
    # krippendorff.alpha needs a numeric value domain even for nominal data, so string
    # categories (e.g. Temporality's labels) must be mapped to integer codes first.
    # Codes are shared across both raters so the same category gets the same code.
    combined = pd.concat([a1, a2], ignore_index=True)
    categories = sorted(combined.dropna().unique())
    mapping = {cat: i for i, cat in enumerate(categories)}
    return a1.map(mapping), a2.map(mapping)


def agreement_for_column(a1, a2, level):
    # Krippendorff's alpha expects one row per rater, one column per item, with NaN for missing.
    if level == "nominal" and a1.dtype == object:
        alpha_a1, alpha_a2 = encode_for_krippendorff(a1, a2)
    else:
        alpha_a1, alpha_a2 = a1, a2
    alpha = krippendorff.alpha(
        reliability_data=[alpha_a1, alpha_a2],
        level_of_measurement=level,
    )

    paired = [(x, y) for x, y in zip(a1, a2) if pd.notna(x) and pd.notna(y)]
    if not paired:
        return alpha, float("nan"), 0
    y1, y2 = zip(*paired)
    if level == "ordinal":
        kappa = cohen_kappa_score(y1, y2, weights="quadratic")
    else:
        kappa = cohen_kappa_score(y1, y2)
    return alpha, kappa, len(paired)


def main():
    a1 = load_annotator(ANNOTATOR1_FILE, "a1")
    a2 = load_annotator(ANNOTATOR2_FILE, "a2")
    merged = a1.merge(a2, on="Sample_ID", validate="one_to_one")

    n_total = len(merged)
    print(f"Sample size: {n_total} items\n")

    all_cols = {**ORDINAL_COLS, **NOMINAL_COLS}
    results = []
    for col, level in all_cols.items():
        v1 = merged[f"{col}__a1"]
        v2 = merged[f"{col}__a2"]
        n_done = int((v1.notna() & v2.notna()).sum())
        if n_done == 0:
            print(f"{col}: 0/{n_total} items labeled by both annotators yet -- skipping.")
            continue
        alpha, kappa, n_paired = agreement_for_column(v1, v2, level)
        kappa_name = "weighted kappa (quadratic)" if level == "ordinal" else "kappa"
        results.append((col, level, n_paired, alpha, kappa_name, kappa))
        print(
            f"{col} ({level}, n={n_paired}/{n_total}): "
            f"Krippendorff's alpha={alpha:.3f}, {kappa_name}={kappa:.3f}"
        )

    if not results:
        print(
            "\nNo rows are fully annotated by both annotators yet. "
            "Fill in Human_Annotation_Sample_v2_Annotator1.xlsx and "
            "Human_Annotation_Sample_v2_Annotator2.xlsx, then rerun this script."
        )


if __name__ == "__main__":
    main()
