import pandas as pd
import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 1. Initialization
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

input_file = 'data/corpus/final_corpus.csv'
output_file = 'data/corpus/final_corpus_annotated.csv'

print(f"📥 Reading data from {input_file}...")
df = pd.read_csv(input_file)

# Check existing progress
if os.path.exists(output_file):
    annotated_df = pd.read_csv(output_file)
    processed_indices = set(annotated_df['Index'].tolist())
    print(f"🔄 Found existing save file. Skipping {len(processed_indices)} already processed rows...")
else:
    processed_indices = set()

# Filter only the rows that haven't been processed yet
df_to_process = df[~df.index.isin(processed_indices)].copy()
df_to_process['Original_Index'] = df_to_process.index

# 2. Define the analysis function (unchanged)
def analyze_sentence(sentence, keyword, year):
    prompt = f"""
    Du bist ein Experte für die deutsche Begriffsgeschichte und politische Linguistik.
    Analysiere den folgenden Satz im Kontext der deutschen Nachkriegszeit (Jahr: {year}).
    Fokus-Wort: "{keyword}"
    
    Satz: "{sentence}"
    
    Bewerte das Fokus-Wort in diesem Satz nach folgenden DREI Kriterien:
    
    1. Nationalistische Intensität (intensity): Tendiert der Kontext eher zur "Blutsgemeinschaft" oder zur "Verfassungsgemeinschaft"? 
       Antworte mit einem ganzzahligen Wert von 1 bis 5. 
       (1 = rein staatsbürgerlich/demokratisch/Verfassungsgemeinschaft, 3 = alltäglich/Bevölkerung, 5 = stark ethnisch/biologisch/Blutsgemeinschaft).
       
    2. Emotionale Valenz (valence): Erscheint das Wort als "Symbol des Stolzes" oder als "Objekt der Reflexion/Mahnung"?
       Antworte mit einem ganzzahligen Wert von 1 bis 5.
       (1 = Objekt der Mahnung/Kritik/historisch belastet, 3 = neutral/deskriptiv, 5 = Symbol des Stolzes/identitätsstiftend).
       
    3. Temporalität (temporality): Wird das Konzept eher als etwas "Vergangenes" oder als etwas "Zukünftiges/Projiziertes" dargestellt?
       Wähle exakt eine dieser drei Kategorien: "historisch/überholt", "gegenwärtig/neutral", "zukünftig/projiziert".
       
    4. Begründung (reasoning): Ein kurzer, präziser Satz zur Begründung.
    
    Antworte AUSSCHLIESSLICH im strengen JSON-Format. Beispiel:
    {{"intensity": 1, "valence": 3, "temporality": "gegenwärtig/neutral", "reasoning": "Bezieht sich auf den souveränen Wähler."}}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs strictly in JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            timeout=15 # Prevents hanging on a single slow request
        )
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'

# Worker function for threading
def process_row(row):
    sentence = row['Hit_Cleaned']
    keyword = row['Keyword']
    year = row['Year']
    index = row['Original_Index']
    
    result_str = analyze_sentence(sentence, keyword, year)
    
    intensity, valence, temporality, reasoning = None, None, None, None
    try:
        result_json = json.loads(result_str)
        if "error" in result_json:
            reasoning = f"API Error: {result_json['error']}"
        else:
            intensity = result_json.get('intensity')
            valence = result_json.get('valence')
            temporality = result_json.get('temporality')
            reasoning = result_json.get('reasoning')
    except json.JSONDecodeError:
        reasoning = "JSON parsing failed"
        
    row_data = row.drop('Original_Index').to_dict()
    row_data['Index'] = index
    row_data['Intensity'] = intensity
    row_data['Valence'] = valence
    row_data['Temporality'] = temporality
    row_data['Reasoning'] = reasoning
    
    return row_data

# 3. Start Multi-threaded Processing
print(f"🚀 Starting accelerated multi-threaded processing for {len(df_to_process)} rows...\n")

# Thread-safe writing setup
lock = threading.Lock()
save_interval = 50 
results_buffer = []

# Number of concurrent workers (10 is a safe sweet-spot to avoid API rate limits)
MAX_WORKERS = 10 

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    futures = {executor.submit(process_row, row): row for _, row in df_to_process.iterrows()}
    
    # Process as they complete
    for future in tqdm(as_completed(futures), total=len(futures), desc="Turbo Annotation Progress"):
        try:
            result = future.result()
            
            with lock:
                results_buffer.append(result)
                
                # Periodically save to disk
                if len(results_buffer) >= save_interval:
                    temp_df = pd.DataFrame(results_buffer)
                    if os.path.exists(output_file):
                        temp_df.to_csv(output_file, mode='a', header=False, index=False)
                    else:
                        temp_df.to_csv(output_file, index=False)
                    results_buffer = [] # Clear buffer
                    
        except Exception as e:
            print(f"Row processing generated an exception: {e}")

# Save any remaining data
if results_buffer:
    temp_df = pd.DataFrame(results_buffer)
    if os.path.exists(output_file):
        temp_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        temp_df.to_csv(output_file, index=False)

print(f"\n🎉 Success! All data processed at high speed and saved to '{output_file}'")