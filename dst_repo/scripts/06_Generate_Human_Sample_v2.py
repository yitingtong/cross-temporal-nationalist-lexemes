import pandas as pd

# Rebuttal follow-up sample: unlike 04_Generate_Human_Sample.py (80 rows, post-war only,
# stratified by Keyword x Period), this draws from the full corpus including the pre-war
# baseline (1918-1945), stratified by Period only (not by Keyword), per the advisor's spec:
# 20 instances per period x 6 periods = 120. The original 80 already-annotated sentences
# are excluded so the new sample is genuinely unseen data for the IAA study.

FULL_CORPUS_FILE = "../data/corpus/final_corpus_annotated.csv"      # 6 periods, 12000 rows, ';'-delimited
MASTER_CORPUS_FILE = "../data/corpus/master_corpus_annotated.csv"    # post-war only, 8000 rows, source of the original 80's Index
EXISTING_SAMPLE_FILE = "../data/human_annotation/Human_Annotation_Sample.xlsx"  # the original 80 already-annotated sentences

N_PER_PERIOD = 20
RANDOM_STATE = 42

full = pd.read_csv(FULL_CORPUS_FILE, sep=";", on_bad_lines="skip", engine="python")
# final_corpus_annotated.csv's "Index" column is not globally unique across periods/keywords
# (it was carried over from per-file row numbers), so dedupe defensively on the composite key.
full = full.drop_duplicates(subset=["Keyword", "Period", "Index"]).reset_index(drop=True)

master = pd.read_csv(MASTER_CORPUS_FILE)
existing_sample = pd.read_excel(EXISTING_SAMPLE_FILE)

# Resolve the original 80's master-corpus "Index" (globally unique there) to the
# (Keyword, Period, Hit_Cleaned, Date) key used to find and exclude them in the full corpus.
existing_resolved = existing_sample.merge(
    master[["Index", "Date", "Hit_Cleaned"]].rename(columns={"Hit_Cleaned": "Hit_Cleaned_master"}),
    on="Index",
)
assert len(existing_resolved) == len(existing_sample), (
    "expected all 80 existing samples to resolve uniquely against master_corpus_annotated.csv"
)

exclude_keys = set(
    zip(
        existing_resolved["Keyword"],
        existing_resolved["Period"],
        existing_resolved["Hit_Cleaned_master"],
        existing_resolved["Date"],
    )
)
full_keys = list(zip(full["Keyword"], full["Period"], full["Hit_Cleaned"], full["Date"]))
pool = full[[k not in exclude_keys for k in full_keys]].copy()

n_excluded = len(full) - len(pool)
print(f"Sampling pool: {len(pool)} rows (excluded {n_excluded} rows matching the original 80)")

# Stratified by Period only, pooled across all 4 keywords within each period.
sample_df = pool.groupby("Period", group_keys=False).sample(n=N_PER_PERIOD, random_state=RANDOM_STATE)
sample_df = sample_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
sample_df.insert(0, "Sample_ID", range(1, len(sample_df) + 1))

assert len(sample_df) == N_PER_PERIOD * 6, f"expected 120 rows, got {len(sample_df)}"

out_cols = ["Sample_ID", "Index", "Keyword", "Period", "Year", "Date", "Hit_Cleaned"]
annotation_cols = [
    "Human_Intensity (1-5)",
    "Human_Valence (1-5)",
    "Human_Temporality",
    "Human_Reasoning (Optional)",
]

# Reference copy (with Sample_ID <-> Index/Keyword/Period lookup) for the researcher's own records.
sample_df[out_cols].to_excel("../data/human_annotation/Human_Annotation_Sample_v2_Reference.xlsx", index=False)

# Two blind copies (identical items, blank annotation columns) for independent annotators.
for annotator in ("Annotator1", "Annotator2"):
    blind_df = sample_df[out_cols].copy()
    for col in annotation_cols:
        blind_df[col] = ""
    blind_df.to_excel(f"../data/human_annotation/Human_Annotation_Sample_v2_{annotator}.xlsx", index=False)

print(f"Generated {len(sample_df)} stratified samples across {sample_df['Period'].nunique()} periods.")
print(sample_df["Period"].value_counts().sort_index())
print(sample_df["Keyword"].value_counts())
