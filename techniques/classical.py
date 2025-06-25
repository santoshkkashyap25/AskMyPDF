# techniques/classical.py
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from utils import preprocess_text_for_vectorization

# --- Model Loading (with Caching) ---
word_embedding_model_cache = {}

def load_word_embedding_model(model_name="glove-wiki-gigaword-50"):
    """Loads a pre-trained word embedding model from gensim-data with caching."""
    if model_name not in word_embedding_model_cache:
        try:
            print(f"Loading word embedding model: {model_name} (this may take a moment)...")
            model = api.load(model_name)
            word_embedding_model_cache[model_name] = model
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading word embedding model '{model_name}': {e}")
            word_embedding_model_cache[model_name] = None
    return word_embedding_model_cache[model_name]

# Pre-load the default model on application start
load_word_embedding_model()

def get_averaged_word_vector(text, model):
    """Calculates the average vector for a text using word embeddings."""
    tokens = nltk.word_tokenize(preprocess_text_for_vectorization(text))
    vectors = [model[token] for token in tokens if token in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_answer_classical_nlp(question, documents_dict, method="tfidf", word_embedding_model_name="glove-wiki-gigaword-50"):
    """
    Finds the best answer using a specified classical NLP method.
    Methods: 'tfidf', 'bow', 'word_embedding'.
    """
    best_answer = {'answer': "No relevant answer found in the documents.", 'confidence': 0.0, 'source_document': None, 'source_chunk': None}
    
    for doc_id, full_text_content in documents_dict.items():
        if not full_text_content or not full_text_content.strip():
            continue
        
        sentences = nltk.sent_tokenize(full_text_content)
        if not sentences:
            continue

        processed_sentences = [preprocess_text_for_vectorization(s) for s in sentences]
        question_processed = preprocess_text_for_vectorization(question)
        
        q_vector, sentence_vectors = None, None
        
        if method in ["tfidf", "bow"]:
            vectorizer = TfidfVectorizer() if method == "tfidf" else CountVectorizer()
            # Ensure we don't fit on an empty vocabulary
            if not any(processed_sentences): continue
            all_vectors = vectorizer.fit_transform(processed_sentences + [question_processed])
            sentence_vectors, q_vector = all_vectors[:-1], all_vectors[-1]
        
        elif method == "word_embedding":
            model = load_word_embedding_model(word_embedding_model_name)
            if model:
                q_vector = get_averaged_word_vector(question_processed, model).reshape(1, -1)
                sentence_vectors = np.array([get_averaged_word_vector(s, model) for s in processed_sentences])
                # Filter out all-zero vectors which can happen with empty processed sentences
                valid_indices = [i for i, vec in enumerate(sentence_vectors) if vec.any()]
                if not valid_indices: continue
                sentence_vectors = sentence_vectors[valid_indices]
                sentences = [sentences[i] for i in valid_indices] # Align original sentences
        
        if q_vector is None or sentence_vectors is None or sentence_vectors.shape[0] == 0:
            continue

        similarities = cosine_similarity(q_vector, sentence_vectors).flatten()
        best_sentence_idx = np.argmax(similarities)
        score = float(similarities[best_sentence_idx]) # Ensure it's a standard float
        
        if score > best_answer['confidence']:
            best_answer.update({
                'answer': sentences[best_sentence_idx],
                'confidence': score,
                'source_document': doc_id,
                'source_chunk': sentences[best_sentence_idx]
            })
            
    return best_answer
