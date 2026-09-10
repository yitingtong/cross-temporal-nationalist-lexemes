# cross-temporal-nationalist-lexemes

Code and derived data for our EMNLP 2026 Findings paper, *"Semantic Shifts of Nationalist Lexemes in German: From Pre-War to Post-War Periods (1918–2024)"*.

The study examines how four German political lexemes — *Volk*, *Nation*, *Staat*, and *Land* — changed in frequency, collocation, and pragmatic framing across a pre-war/Nazi-era baseline (1918–1945) and the post-war period (1945–2024), using corpora from the [DWDS](https://www.dwds.de/) (Digitales Wörterbuch der deutschen Sprache). Frequency and collocation trends were produced directly with the DWDS web interface; the sentence-level analysis (nationalist intensity, emotional valence, temporality) was produced with LLM-assisted annotation, validated against human annotators.

## Repository structure

```
data/
  raw/               24 CSVs exported from DWDS (Gegenwartskorpora + Zeitungskorpus):
                      one file per lexeme x historical period (Volk/Nation/Staat/Land
                      x 1918-1932, 1933-1945, 1946-1969, 1970-1989, 1990-2004, 2005-2024)
  corpus/            Assembled corpora used by the annotation/analysis scripts
                      - final_corpus.csv            all 6 periods, unannotated (12,000 sampled sentences)
                      - final_corpus_annotated.csv   all 6 periods, with LLM annotations
                      - master_corpus.csv            post-war periods only, unannotated (8,000 sentences)
                      - master_corpus_annotated.csv  post-war periods only, with LLM annotations
  human_annotation/  Human evaluation spreadsheets (see scripts 04-09)
scripts/             Pipeline scripts and notebooks, numbered in the order they're meant to run
figures/             Figures from the paper (figure1-6, matching the paper's numbering) plus the
                      DWDS frequency/collocation exports
```

## Pipeline

Scripts `02`, `04`, `05` and notebooks `01`, `03` are meant to be run from the **repository root**; scripts `06`-`09` are meant to be run from inside `scripts/` (they use `../data/...` paths). All data paths in the scripts already point at the `data/` layout above.

1. `scripts/01_LLM_Annotation_Test.ipynb` — small-scale test of the LLM annotation prompt (not part of the main pipeline).
2. `scripts/02_Full_Corpus_Annotation.py` — runs the LLM-assisted annotation (nationalist intensity, emotional valence, temporality) over `data/corpus/final_corpus.csv` using the OpenAI API, producing `data/corpus/final_corpus_annotated.csv`. Requires an `OPENAI_API_KEY` (see below).
3. `scripts/03_Data_Analysis_Visualization.ipynb` — frequency/intensity/valence/temporality plots from the annotated corpus (paper Figures 4-6).
4. `scripts/04_Generate_Human_Sample.py` — draws the original 80-sentence stratified human-evaluation sample (post-war only) from `data/corpus/master_corpus_annotated.csv`.
5. `scripts/05_Calculate_Reliability.py` — LLM-vs-human agreement (Spearman / Cohen's kappa) on the 80-sentence sample.
6. `scripts/06_Generate_Human_Sample_v2.py` — draws the follow-up 120-sentence sample (all 6 periods, including pre-war) for the extended inter-annotator agreement study.
7. `scripts/07_Add_LLM_Comparison.py` — attaches the existing LLM annotations to the 120-sentence reference sheet for later comparison.
8. `scripts/08_Calculate_IAA_v2.py` — inter-annotator agreement (Krippendorff's alpha, Cohen's kappa) between the two human annotators on the 120-sentence sample.
9. `scripts/09_Calculate_Reliability_v2.py` — LLM-vs-human agreement on the 120-sentence sample, using the same methodology as `05` for direct comparability.

Frequency curves (Figures 1-2) and collocation profiles (Figure 3) were generated directly through the DWDS web interface (frequency tool and DiaCollo) rather than by a script in this repository; the exported figures are included under `figures/`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your own OPENAI_API_KEY
```

`scripts/02_Full_Corpus_Annotation.py` and `scripts/01_LLM_Annotation_Test.ipynb` call the OpenAI API (`gpt-5-mini` in the paper) and require `OPENAI_API_KEY` to be set (loaded from `.env` via `python-dotenv`). No other script needs API access.

## Data notes

- `data/raw/` CSVs are directly exported DWDS query results, standardized to a common 5-column schema (`No.`, `Date`, `Genre`, `Bibl.`, `Hit`) across periods.
- `data/corpus/final_corpus.csv` / `final_corpus_annotated.csv` cover all six historical periods (1918-2024, ~12,000 sampled sentences); `master_corpus.csv` / `master_corpus_annotated.csv` cover only the four post-war periods (1945-2024, ~8,000 sampled sentences) and are the basis for the original 80-sentence human evaluation sample.
- `data/human_annotation/Human_Annotation_Sample.xlsx` is the original 80-sentence sample (post-war only); the `_v2_*` files are the follow-up 120-sentence sample (all six periods) used for the extended inter-annotator agreement study, with blind copies for two independent annotators (`_Annotator1`/`_Annotator2`) and a reference copy with LLM labels attached (`_Reference_with_LLM`).

## Citation

If you use this code or data, please cite our EMNLP 2026 Findings paper (citation details to be added upon publication).
