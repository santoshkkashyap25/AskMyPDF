import os
import csv
import json
import argparse
import google.generativeai as genai
from time import sleep
from typing import List, Dict, Any

# 1. Get your key from https://aistudio.google.com/app/apikey
# 2. Set it in your terminal: export GOOGLE_API_KEY='your-key'
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except TypeError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please get a key from Google AI Studio and set it using: export GOOGLE_API_KEY='your-api-key'")
    exit(1)

MODEL = "gemini-1.5-flash-latest"

# --- Functions ---

def generate_qa_pairs(case_text: str, case_title: str, case_id: str) -> List[Dict[str, Any]]:
    print(f"  - Generating QA pairs for {case_id} with Google Gemini...")
    
    prompt = f"""
    Based on the following legal case text, generate exactly 2 distinct question-and-answer pairs.
    The 'answers' must be an EXACT verbatim quote from the provided text.
    Format the output as a single valid JSON object with one key, "qa_pairs", which contains a list of the question-answer objects.

    Here is the required JSON structure for each object in the list:
    {{
      "id": "A unique identifier for the QA pair (e.g., {case_id}_QA1)",
      "title": "{case_title}",
      "context": "The full case text...",
      "question": "A relevant legal question that can be answered from the text.",
      "answers": {{
        "text": ["The exact answer string, copied directly from the context."],
        "answer_start": [The integer start index of the answer within the context.]
      }}
    }}

    Legal Case Text:
    ---
    {case_text}
    ---
    """

    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel(MODEL)
        
        # Configure the model to output JSON
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=2048,
            temperature=0.4
        )

        # Call the API
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # The response.text should be a JSON string
        generated_data = json.loads(response.text)

        # The model should return a dict like {"qa_pairs": [...]}. We extract the list.
        if isinstance(generated_data, dict):
            qa_list = generated_data.get("qa_pairs")
            if not isinstance(qa_list, list):
                print(f"  - Gemini Warning for {case_id}: JSON response did not contain a 'qa_pairs' list.")
                return []
        else:
            print(f"  - Gemini Warning for {case_id}: Response was not a JSON object as expected.")
            return []

        # --- Validate and format each generated pair ---
        valid_pairs = []
        for i, qa in enumerate(qa_list, 1):
            try:
                # Ensure all required keys exist
                question = qa['question']
                answer_text = qa['answers']['text'][0]

                # Critical validation: Find the exact answer span in the original text
                start_index = case_text.find(answer_text)

                if start_index == -1:
                    print(f"  - Validation ERROR for {case_id}: Generated answer was not found in the original text. Skipping.")
                    print(f"    - Answer not found: '{answer_text[:100]}...'")
                    continue  # Skip this malformed pair

                # This pair is valid, format it correctly.
                valid_pairs.append({
                    "id": f"{case_id}_QA{i}",
                    "title": case_title,
                    "context": case_text,
                    "question": question,
                    "answers": {
                        "text": [answer_text],
                        "answer_start": [start_index]
                    }
                })
                print(f"  - Successfully validated QA pair {i} for {case_id}.")

            except (KeyError, IndexError, TypeError) as e:
                print(f"  - Validation ERROR for {case_id}: QA pair has missing or malformed data. Skipping. Details: {e}")

        return valid_pairs

    except json.JSONDecodeError:
        print(f"  - Gemini ERROR for {case_id}: Failed to decode JSON from the model's response.")
        return []
    except Exception as e:
        print(f"  - An unexpected error occurred during API call for {case_id}: {e}")
        return []

def main(input_csv: str, output_json: str):
    print(f"Starting processing...")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_json}")

    all_qa_data = []

    try:
        with open(input_csv, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_num, row in enumerate(reader, 1):
                case_id = row.get('case_id')
                case_text = row.get('case_text')
                case_title = row.get('case_title')

                if not all([case_id, case_text, case_title]):
                    print(f"Skipping row {row_num} due to missing data (id, text, or title).")
                    continue
                
                print(f"\nProcessing row {row_num}: Case ID {case_id}")

                generated_pairs = generate_qa_pairs(
                    case_text=case_text,
                    case_title=case_title,
                    case_id=case_id
                )

                if generated_pairs:
                    all_qa_data.extend(generated_pairs)

                # Rate limiting to avoid overwhelming the API.
                # The free tier has a limit of 60 requests per minute.
                sleep(1.5) 

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv}'")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
    finally:
        if all_qa_data:
            print(f"\nProcessing complete. Saving {len(all_qa_data)} QA pairs to {output_json}...")
            try:
                # The final SQuAD format has a 'data' key holding the list of all cases.
                squad_formatted_output = {"data": all_qa_data}
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(squad_formatted_output, f, indent=2, ensure_ascii=False)
                print("File saved successfully.")
            except Exception as e:
                print(f"Error saving final JSON file: {e}")
        else:
            print("\nProcessing complete. No valid QA pairs were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a SQuAD-style legal QA dataset from a CSV using Google Gemini."
    )
    parser.add_argument(
        "--input",
        default="/home/shtlp_0103/Practice_Python/legal_text_classification.csv",
        help="Path to the input CSV file. (default: legal_cases.csv)"
    )
    parser.add_argument(
        "--output",
        default="legal_qa_squad_gemini.json",
        help="Path for the output JSON file. (default: legal_qa_squad_gemini.json)"
    )
    
    args = parser.parse_args()
    
    main(input_csv=args.input, output_json=args.output)