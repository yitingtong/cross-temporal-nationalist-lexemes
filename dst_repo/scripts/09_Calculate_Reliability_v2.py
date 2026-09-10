import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

# LLM vs. Human reliability for the 120-item rebuttal sample (all 6 periods, incl. pre-war),
# using the same methodology as 05_Calculate_Reliability.py (Spearman for the ordinal
# Intensity/Valence scales, Cohen's Kappa for the nominal Temporality label) so the numbers
# are directly comparable to the original 80-sample figures already reported in the paper.
# Does NOT touch 05_Calculate_Reliability.py or its inputs -- that result stays reproducible
# as-is. Reports LLM agreement against each annotator separately (not pooled), since the two
# annotators haven't been shown to agree with each other yet at this point in the pipeline.

REFERENCE_WITH_LLM_FILE = "../data/human_annotation/Human_Annotation_Sample_v2_Reference_with_LLM.xlsx"
ANNOTATOR_FILES = {
    "Annotator1": "../data/human_annotation/Human_Annotation_Sample_v2_Annotator1.xlsx",
    "Annotator2": "../data/human_annotation/Human_Annotation_Sample_v2_Annotator2.xlsx",
}


def compute_reliability(df_merged, annotator_name):
    df_merged = df_merged.copy()
    df_merged["LLM_Intensity (1-5)"] = pd.to_numeric(df_merged["LLM_Intensity (1-5)"], errors="coerce")
    df_merged["Human_Intensity (1-5)"] = pd.to_numeric(df_merged["Human_Intensity (1-5)"], errors="coerce")
    df_merged["LLM_Valence (1-5)"] = pd.to_numeric(df_merged["LLM_Valence (1-5)"], errors="coerce")
    df_merged["Human_Valence (1-5)"] = pd.to_numeric(df_merged["Human_Valence (1-5)"], errors="coerce")
    df_merged["LLM_Temporality"] = df_merged["LLM_Temporality"].astype(str).str.strip().str.lower()
    df_merged["Human_Temporality"] = df_merged["Human_Temporality"].astype(str).str.strip().str.lower()

    df_merged = df_merged.dropna(subset=["Human_Intensity (1-5)", "Human_Valence (1-5)", "Human_Temporality"])
    n = len(df_merged)
    if n == 0:
        print(f"{annotator_name}: 0/120 items labeled yet -- skipping.")
        return

    rho_intensity, p_intensity = spearmanr(df_merged["LLM_Intensity (1-5)"], df_merged["Human_Intensity (1-5)"])
    rho_valence, p_valence = spearmanr(df_merged["LLM_Valence (1-5)"], df_merged["Human_Valence (1-5)"])
    kappa_temporality = cohen_kappa_score(df_merged["LLM_Temporality"], df_merged["Human_Temporality"])

    print(f"--- LLM vs. {annotator_name} (n={n}/120) ---")
    print(f"Nationalist Intensity (Spearman rho): {rho_intensity:.3f} (p={p_intensity:.3f})")
    print(f"Emotional Valence     (Spearman rho): {rho_valence:.3f} (p={p_valence:.3f})")
    print(f"Temporality           (Cohen's kappa): {kappa_temporality:.3f}")
    print()


def main():
    ref_llm = pd.read_excel(REFERENCE_WITH_LLM_FILE)
    for annotator_name, path in ANNOTATOR_FILES.items():
        human_df = pd.read_excel(path)
        merged = ref_llm.merge(human_df[["Sample_ID", "Human_Intensity (1-5)", "Human_Valence (1-5)", "Human_Temporality"]], on="Sample_ID")
        compute_reliability(merged, annotator_name)


if __name__ == "__main__":
    main()
