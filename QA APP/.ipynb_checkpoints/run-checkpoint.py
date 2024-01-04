from flask import Flask, render_template, request
import pdfplumber
import nltk
import re
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import remove_stopwords

app = Flask(__name__)

# Default empty values for pdf_txt and cleaned_sentences
pdf_txt = ""
cleaned_sentences = []

nltk.download('punkt')

# GloVe model
glove_model_path = 'glovemodel.mod'  # Update this path based on your setup

# Check if the GloVe model file exists
if not os.path.exists(glove_model_path):
    # If the model file doesn't exist, download and save the model
    glove_model = api.load('glove-twitter-25')
    glove_model.save(glove_model_path)
    print("GloVe Model Saved")
else:
    # If the model file exists, load the model
    glove_model = gensim.models.KeyedVectors.load(glove_model_path)
    print("GloVe Model Successfully loaded")

glove_embedding_size = len(glove_model['pc'])  # random word

# TF-IDF vectorization
stop_words = nltk.corpus.stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = None

# Clean sentences
def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence

# Function to get GloVe embedding for a phrase
def get_phrase_embedding(phrase, model):
    words = phrase.split()
    # Take the average of word embeddings for the phrase
    phrase_embedding = sum(model[word] for word in words if word in model) / len(words)
    return phrase_embedding

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    global pdf_txt, cleaned_sentences, tfidf_matrix
    
    if request.method == 'POST':
        # Process uploaded PDF file
        pdf_file = request.files['pdf_file']
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            # Extract text from the uploaded PDF
            pdf = pdfplumber.open(pdf_file)
            # Concatenate text from all pages
            pdf_txt = ""
            for page in pdf.pages:
                pdf_txt += page.extract_text()
            
            pdf.close()

            # Tokenize and preprocess sentences
            tokens = nltk.sent_tokenize(pdf_txt)
            cleaned_sentences = [clean_sentence(row, stopwords=True) for row in tokens]

            # Update TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)

        user_question = request.form['user_question']

        # Answer using TF-IDF
        cleaned_user_question_tfidf = clean_sentence(user_question, stopwords=True)
        user_question_vector_tfidf = vectorizer.transform([cleaned_user_question_tfidf])
        tfidf_similarity_scores = cosine_similarity(user_question_vector_tfidf, tfidf_matrix)
        tfidf_index_sim = tfidf_similarity_scores.argmax()
        tfidf_answer = cleaned_sentences[tfidf_index_sim]

        # Answer using GloVe
        cleaned_user_question_glove = clean_sentence(user_question, stopwords=True)
        user_question_embedding_glove = get_phrase_embedding(cleaned_user_question_glove, glove_model)
        glove_similarity_scores = cosine_similarity([user_question_embedding_glove], [get_phrase_embedding(sent, glove_model) for sent in cleaned_sentences])
        glove_index_sim = glove_similarity_scores.argmax()
        glove_answer = cleaned_sentences[glove_index_sim]

        return render_template('result.html', tfidf_answer=tfidf_answer, glove_answer=glove_answer)

if __name__ == '__main__':
    app.run(debug=True)
