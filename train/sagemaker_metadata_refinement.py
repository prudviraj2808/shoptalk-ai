import json
import boto3
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

# --- CONFIGURATION ---
ENDPOINT_NAME = "jumpstart-dft-meta-textgeneration-l-20260218-063657"
REGION = "us-east-1"
INPUT_FILE = "training_data/full_metadata.jsonl"
OUTPUT_FILE = "training_data/refined_sagemaker_metadata.jsonl"
MAX_WORKERS = 30 
TEST_LIMIT = None  # Set to None for full run

config = Config(
    read_timeout=300, 
    connect_timeout=60, 
    retries={"max_attempts": 3}
)

client = boto3.client("sagemaker-runtime", region_name=REGION, config=config)

def clean_extra_chatter(text):
    """Removes all conversational prefixes to isolate the sentence."""
    patterns = [
        r"^for the product:?\s*",
        r"^based on the .* information:?\s*",
        r"^the visual description is:?\s*",
        r"^refined caption:?\s*",
        r"^visual sentence:?\s*",
        r"^description:?\s*",
        r"^sentence:?\s*",
        r"^sure, here is:?\s*"
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip().strip('"').strip("'")

def process_record(record):
    meta = record.get('metadata', {})
    exclude_keys = {'item_id', 'image_path', 'csv_path', 'main_image_id', 'other_image_id', 'height', 'width'}
    
    input_data = []
    if record.get('text'):
        input_data.append(f"Context_Title: {record.get('text')}")
    
    for key, value in meta.items():
        if key not in exclude_keys and value:
            val_str = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            if val_str.strip():
                label = f"Context_{key.replace('_', ' ').title()}"
                input_data.append(f"{label}: {val_str}")

    data_context = "\n".join(input_data)

    # UPDATED PROMPT: Added strict length and catalogue constraints
    prompt = (
        f"<s>[INST] <<SYS>>\nYou are a professional Catalogue Captioner. "
        f"Your task is to construct ONE concise visual sentence using ONLY the vocabulary present in the Context Data.\n"
        f"STRICT RULES:\n"
        f"1. LENGTH: Maximum 40 words total.\n"
        f"2. EXTRACTIVE: Use ONLY words found in the provided Context Data. No external adjectives.\n"
        f"3. VISUAL: Exclude logistical info (Warranty, Service, Shipping, Buttons, Ports).\n"
        f"4. NO preamble. Start the sentence immediately.\n<</SYS>>\n\n"
        f"CONTEXT DATA:\n{data_context}\n\n"
        f"COMMAND: Synthesize the context into a dense, catalogue-style visual sentence under 40 words.\n"
        f"Visual Sentence: [/INST] A"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 65, # Tightened to meet 65 token limit for MobileCLIP2-S2
            "temperature": 0.01,
            "top_p": 0.01,
            "return_full_text": False,
            "stop": ["\n", "[/INST]", "For the", "Based on"]
        }
    }

    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read().decode("utf-8"))

        if isinstance(result, list) and len(result) > 0:
            caption = result[0].get("generated_text", "").strip()
        elif isinstance(result, dict):
            caption = result.get("generated_text", "").strip()
        else:
            caption = str(result).strip()

        # Clean and restore forced 'A'
        caption = "A " + clean_extra_chatter(caption)
        caption = caption.rstrip(":").strip()

        record["refined_caption"] = caption
        return record

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    records = []
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    if TEST_LIMIT:
        print(f"--- TEST MODE: Processing {TEST_LIMIT} records ---")
        records = records[:TEST_LIMIT]

    processed_results = []
    # Using 30 workers as specified
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, rec) for rec in records]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                # Removed full print for large runs, but kept ID logging for verification
                processed_results.append(res)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in processed_results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Successfully processed {len(processed_results)} records.")

if __name__ == "__main__":
    main()