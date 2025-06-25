# utils.py
import io
import re
import PyPDF2
import pdfplumber
import nltk

# --- NLTK Data Check ---
def download_nltk_data_if_needed():
    """Checks for and downloads required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
        print("'punkt' downloaded.")
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'stopwords'...")
        nltk.download('stopwords')
        print("'stopwords' downloaded.")

# Run the check once on startup
download_nltk_data_if_needed()

# --- Text Extraction ---
def extract_text_from_pdf(file_bytes, filename="<file>"):
    """
    Extracts text from a PDF file's bytes.
    Tries pdfplumber first, falls back to PyPDF2.
    """
    text = ""
    try:
        # Try with pdfplumber first as it often gives better layout parsing
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        if text.strip():
            print(f"Successfully extracted text from '{filename}' using pdfplumber.")
            return text
    except Exception as e:
        print(f"pdfplumber failed for '{filename}': {e}. Trying PyPDF2.")

    # Fallback to PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        if text.strip():
            print(f"Successfully extracted text from '{filename}' using PyPDF2.")
    except Exception as e:
        print(f"PyPDF2 also failed for '{filename}': {e}")
        return "" # Return empty if both fail
        
    return text

# --- Text Preprocessing ---
def preprocess_text_for_vectorization(text):
    """
    Cleans and tokenizes text for classical NLP models.
    - Lowercases
    - Removes punctuation
    - Removes stopwords
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)
