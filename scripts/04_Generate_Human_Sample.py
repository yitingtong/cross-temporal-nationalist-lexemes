import pandas as pd

input_file = 'data/corpus/master_corpus_annotated.csv'
output_file = 'data/human_annotation/Human_Annotation_Sample.xlsx'
df = pd.read_csv(input_file)

# Stratified Random Sampling
# We have 4 keywords and 4 periods = 16 strata. 
# We sample exactly 5 rows per stratum to get a perfectly balanced 80-row sample.
# random_state=42 ensures reproducibility (the exact same 80 rows will be picked every time)
sample_df = df.groupby(['Keyword', 'Period']).sample(n=5, random_state=42).copy()
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

human_coding_df = sample_df[['Index', 'Keyword', 'Period', 'Year', 'Hit_Cleaned']].copy()
human_coding_df['Human_Intensity (1-5)'] = ""
human_coding_df['Human_Valence (1-5)'] = ""
human_coding_df['Human_Temporality'] = ""
human_coding_df['Human_Reasoning (Optional)'] = ""

human_coding_df.to_excel(output_file, index=False)

print(f"✅ Success! Stratified random sample of {len(human_coding_df)} rows generated.")