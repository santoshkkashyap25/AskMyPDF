from flask import Flask, render_template, request
import pdfplumber
import nltk
import os
import re
import numpy
import gensim
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora

app = Flask(__name__)

def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence

# Function to convert sentences to Bag-of-Words representation
def sentences_to_bow(sentences, dictionary):
    sentence_words = [[word for word in document.split()] for document in sentences]
    bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
    return bow_corpus

def get_cleaned_sentences(tokens, stopwords=False):
    cleaned_sentences = []
    for row in tokens:
        cleaned = clean_sentence(row, stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences


def retrieve_and_print_faq_answer(question_embedding, sentence_embeddings, sentences):
    max_sim = -1
    index_sim = -1
    for index, embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(embedding, question_embedding)[0][0]
        if sim > max_sim:
            max_sim = sim
            index_sim = index
    return index_sim

def get_word_vec(word, model):
    samp = model['pc']
    vec = [0] * len(samp)
    try:
        vec = model[word]
    except:
        vec = [0] * len(samp)
    return vec

def get_phrase_embedding(phrase, embedding_model):
    samp = get_word_vec('computer', embedding_model)
    vec = numpy.array([0] * len(samp))
    den = 0
    for word in phrase.split():
        den = den + 1
        vec = vec + numpy.array(get_word_vec(word, embedding_model))
    return vec.reshape(1, -1)


def load_model(model_path, model_name):
    if os.path.exists(model_path):
        model = gensim.models.KeyedVectors.load(model_path)
        print(f"{model_name} Model Successfully loaded")
    else:
        model = api.load(model_name)
        model.save(model_path)
        print(f"{model_name} Model Saved")
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        pdf = pdfplumber.open(pdf_file)
        pdf_txt = ""
        for page_number in range(len(pdf.pages)):
            page = pdf.pages[page_number]
            pdf_txt += page.extract_text()
        pdf.close()

        nltk.download('punkt')
        tokens = nltk.sent_tokenize(pdf_txt)

        cleaned_sentences_with_stopwords = get_cleaned_sentences(tokens, stopwords=False)
        sentences = cleaned_sentences_with_stopwords

        # TF-IDF Approach
        stop_words = stopwords.words('english')
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cleaned_sentences_with_stopwords)
        user_question = request.form['user_question']
        cleaned_user_question = ' '.join([word for word in nltk.word_tokenize(user_question) if word.isalnum() and word not in stop_words])
        user_question_vector = vectorizer.transform([cleaned_user_question])
        similarity_scores = cosine_similarity(user_question_vector, tfidf_matrix)
        index_sim_tfidf = similarity_scores.argmax()
        answer_tfidf = cleaned_sentences_with_stopwords[index_sim_tfidf]

        # GloVe Approach
        glove_model = load_model('D:\\NLP\\Question Answering\\QA APP\\glovemodel.mod', 'glove-twitter-25')
        glove_embedding_size = len(glove_model['pc'])
        sent_embeddings_glove = [get_phrase_embedding(sent, glove_model) for sent in cleaned_sentences_with_stopwords]
        question_embedding_glove = get_phrase_embedding(user_question, glove_model)
        index_sim_glove = retrieve_and_print_faq_answer(question_embedding_glove, sent_embeddings_glove, cleaned_sentences_with_stopwords)
        answer_glove = cleaned_sentences_with_stopwords[index_sim_glove]

        # Bag-of-Words Approach
        dictionary = corpora.Dictionary([sentence.split() for sentence in cleaned_sentences_with_stopwords])
        bow_corpus = sentences_to_bow(cleaned_sentences_with_stopwords, dictionary)
        question_bow = dictionary.doc2bow(clean_sentence(user_question, stopwords=False).split())
        index_sim_bow = retrieve_and_print_faq_answer(question_bow, bow_corpus, cleaned_sentences_with_stopwords)
        answer_bow = cleaned_sentences_with_stopwords[index_sim_bow]

        return render_template('result.html', answer_tfidf=answer_tfidf, answer_glove=answer_glove, answer_bow=answer_bow)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
