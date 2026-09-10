import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr

llm_file = 'data/corpus/master_corpus_annotated.csv'
human_file = 'data/human_annotation/Human_Annotation_Sample.xlsx'

df_llm = pd.read_csv(llm_file)
df_human = pd.read_excel(human_file)

df_llm_subset = df_llm[['Index', 'Intensity', 'Valence', 'Temporality']]
df_merged = pd.merge(df_human, df_llm_subset, on='Index', how='inner')

df_merged['Intensity'] = pd.to_numeric(df_merged['Intensity'], errors='coerce')
df_merged['Human_Intensity (1-5)'] = pd.to_numeric(df_merged['Human_Intensity (1-5)'], errors='coerce')
df_merged['Valence'] = pd.to_numeric(df_merged['Valence'], errors='coerce')
df_merged['Human_Valence (1-5)'] = pd.to_numeric(df_merged['Human_Valence (1-5)'], errors='coerce')

df_merged['Temporality'] = df_merged['Temporality'].astype(str).str.strip().str.lower()
df_merged['Human_Temporality'] = df_merged['Human_Temporality'].astype(str).str.strip().str.lower()

df_merged = df_merged.dropna(subset=['Human_Intensity (1-5)', 'Human_Valence (1-5)', 'Human_Temporality'])

# Intensity & Valence：Spearman 相关（人 vs GPT，数值量表）
rho_intensity, p_intensity = spearmanr(df_merged['Intensity'], df_merged['Human_Intensity (1-5)'])
rho_valence, p_valence = spearmanr(df_merged['Valence'], df_merged['Human_Valence (1-5)'])

# Temporality：Cohen's Kappa（分类变量，保留）
kappa_temporality = cohen_kappa_score(df_merged['Temporality'], df_merged['Human_Temporality'])

print("=" * 50)
print(f"Nationalist Intensity (Spearman ρ): {rho_intensity:.3f}  (p={p_intensity:.3f})")
print(f"Emotional Valence    (Spearman ρ): {rho_valence:.3f}  (p={p_valence:.3f})")
print(f"Temporality          (Cohen's κ):  {kappa_temporality:.3f}")
print("=" * 50)