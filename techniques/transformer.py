# techniques/transformer.py
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

qa_pipeline_cache = {}

def load_qa_pipeline(model_name="deepset/tinyroberta-squad2"):
    if model_name not in qa_pipeline_cache:
        try:
            print(f"Loading QA pipeline for model: {model_name} (this may take a moment)...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            qa_pipeline_cache[model_name] = pipeline(
                "question-answering", model=model, tokenizer=tokenizer, device=-1
            )
            qa_pipeline_cache[model_name](
                question="What is AI?", context="AI is artificial intelligence."
            )
            print("QA pipeline loaded and warmed up on CPU.")
        except Exception as e:
            print(f"Error loading QA pipeline '{model_name}': {e}")
            qa_pipeline_cache[model_name] = None
    return qa_pipeline_cache[model_name]

load_qa_pipeline()


def chunk_text(text, tokenizer, max_chunk_size=384, overlap_size=64):
    """Splits text into overlapping chunks suitable for the model's context window."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_chunk_size - overlap_size):
        chunk_tokens = tokens[i : i + max_chunk_size]
        # Decode tokens back to a string
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def get_best_answer_transformer(question, documents_dict, confidence_threshold=0.05):
    qa_pipeline = load_qa_pipeline()
    if not qa_pipeline:
        return {'answer': "Transformer model is currently unavailable.", 'confidence': 0.0, 'source_document': None, 'source_chunk': None}

    all_possible_answers = []
    tokenizer = qa_pipeline.tokenizer

    for doc_id, full_text_content in documents_dict.items():
        if not full_text_content or not full_text_content.strip():
            continue
        
        if len(tokenizer.encode(question, full_text_content)) < tokenizer.model_max_length:
            contexts = [full_text_content] # Process the whole document as one context
        else:
            print(f"Document '{doc_id}' is too long, chunking...")
            contexts = chunk_text(full_text_content, tokenizer)
        
        for context_chunk in contexts:
            if not context_chunk.strip(): continue
            try:
                result = qa_pipeline(question=question, context=context_chunk)
                if result['score'] > confidence_threshold:
                    all_possible_answers.append({
                        'answer': result['answer'], 
                        'confidence': result['score'], 
                        'source_document': doc_id, 
                        'source_chunk': context_chunk
                    })
            except Exception as e:
                print(f"Error processing chunk from '{doc_id}': {e}")
    
    if not all_possible_answers:
        return {'answer': "No answer found with sufficient confidence.", 'confidence': 0.0, 'source_document': None, 'source_chunk': None}
        
    # Sort and return the single best answer from all documents and chunks
    best_answer = sorted(all_possible_answers, key=lambda x: x['confidence'], reverse=True)[0]
    return best_answer
