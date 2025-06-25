# app.py
import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from utils import extract_text_from_pdf
from techniques.classical import get_answer_classical_nlp
from techniques.transformer import get_best_answer_transformer

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_secure_default_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page where users can upload documents,
    enter text, and ask a question.
    """
    if request.method == 'POST':
        # --- Form Data Handling ---
        question = request.form.get('question')
        direct_text = request.form.get('direct_text')
        techniques = request.form.getlist('techniques')
        uploaded_files = request.files.getlist('pdf_files')

        # --- Input Validation ---
        if not question:
            flash('Please provide a question.', 'error')
            return redirect(request.url)

        if not techniques:
            flash('Please select at least one technique to use.', 'error')
            return redirect(request.url)
        
        if not any(f.filename for f in uploaded_files) and not direct_text.strip():
            flash('Please upload at least one PDF or provide some text.', 'error')
            return redirect(request.url)

        # --- Document Processing ---
        all_documents_text = {}
        
        # Process direct text input
        if direct_text.strip():
            all_documents_text['direct_text_input'] = direct_text
        
        # Process uploaded PDF files
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_bytes = file.read()
                
                extracted_text = extract_text_from_pdf(file_bytes, filename)
                
                if extracted_text:
                    all_documents_text[filename] = extracted_text
                else:
                    flash(f"Could not extract text from '{filename}'. It might be empty or corrupted.", 'warning')

        if not all_documents_text:
            flash('Failed to process any of the provided documents.', 'error')
            return redirect(request.url)
            
        # --- Answer Retrieval ---
        results = {}
        active_techniques = []
        
        if 'compare_all' in techniques:
            active_techniques = ['tfidf', 'bow', 'word_embedding', 'transformer']
        else:
            active_techniques = techniques

        for tech in active_techniques:
            if tech == 'transformer':
                 results['transformer'] = get_best_answer_transformer(question, all_documents_text)
            else:
                 results[tech] = get_answer_classical_nlp(question, all_documents_text, method=tech)
        
        return render_template('results.html', question=question, results=results, techniques=active_techniques)

    return render_template('index.html')

if __name__ == '__main__':
    # Using threaded=False is important for some NLP models to avoid issues
    app.run(debug=True, threaded=False)
