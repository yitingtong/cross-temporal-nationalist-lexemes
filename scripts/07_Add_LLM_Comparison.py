import pandas as pd

# Adds the existing LLM annotation (from final_corpus_annotated.csv) for each of the 120
# rebuttal sample sentences to a copy of the reference file, so human vs. LLM labels can be
# compared once the two annotators return their sheets. Writes a NEW file — never touches
# Human_Annotation_Sample_v2_Annotator1/2.xlsx, which must stay blind for the annotators.

REFERENCE_FILE = "../data/human_annotation/Human_Annotation_Sample_v2_Reference.xlsx"
FULL_CORPUS_FILE = "../data/corpus/final_corpus_annotated.csv"
OUTPUT_FILE = "../data/human_annotation/Human_Annotation_Sample_v2_Reference_with_LLM.xlsx"

reference = pd.read_excel(REFERENCE_FILE)

full = pd.read_csv(FULL_CORPUS_FILE, sep=";", on_bad_lines="skip", engine="python")
# Same dedup as 06_Generate_Human_Sample_v2.py: Index is only unique within (Keyword, Period).
full = full.drop_duplicates(subset=["Keyword", "Period", "Index"])

llm_cols = full[["Keyword", "Period", "Index", "Intensity", "Valence", "Temporality", "Reasoning"]].rename(
    columns={
        "Intensity": "LLM_Intensity (1-5)",
        "Valence": "LLM_Valence (1-5)",
        "Temporality": "LLM_Temporality",
        "Reasoning": "LLM_Reasoning",
    }
)

merged = reference.merge(llm_cols, on=["Keyword", "Period", "Index"], how="left")

assert len(merged) == len(reference), "row count changed during merge — join key is not unique"
n_missing = merged["LLM_Intensity (1-5)"].isna().sum()
if n_missing:
    print(f"WARNING: {n_missing} of {len(merged)} rows had no matching LLM annotation")

merged.to_excel(OUTPUT_FILE, index=False)
print(f"Wrote {OUTPUT_FILE} ({len(merged)} rows, {n_missing} missing LLM annotations)")
