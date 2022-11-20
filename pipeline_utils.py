#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm')
import string
import pickle

import gensim
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ! conda install -c conda-forge sentence-transformers

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from summa import summarizer

from gensim.models import Word2Vec

# ! pip install transformers -q
# ! pip install simpletransformers wandb pytorch-lightning
# ! pip install -U transformers torch sentencepiece
# ! pip install -U summa


# # $\textbf{All Pre-processing Functions on Top}$

# In[4]:


def read_data(path):
    df = pd.read_pickle(path,compression = 'zstd')
    df = df.dropna(how="any")
    return df

def tokenize_and_remove_stopwords(X):
    stop_words = set(stopwords.words('english'))
    return X['tokens'].apply(lambda x: [word for word in x if word not in stop_words])


def preprocess(df):

    def remove_space(match_obj):
        if match_obj.group() is not None:
            return match_obj.group().replace(' ','|')

    df['Text'] = df['Text'].str.replace(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', 'URL', regex=True)


    # remove '|' so we can use it as a unique space seperator for cases
    df['Text'] = df['Text'].str.replace(r'|', '', regex=True)


    df['Text'] = df['Text'].str.replace(r'U\. S\. C\.', 'U.S.C.', regex=True)
    df['Text'] = df['Text'].str.replace(r'U\. S\.', 'U.S.', regex=True)
    df['Text'] = df['Text'].str.replace(r'No\. ', 'No.', regex=True)
    r"F\.( \d+-?\w*,?)+( \([A-Z]*\d* ?\d*\),?)*"
    # df['Text'] = df['Text'].str.replace(r"F\.( \d+-?\w*,?)+( \([A-Z]*\d* ?\d*\),?)*", remove_space, regex=True)
    text = []
    for element in df['Text']:
        element = re.sub(r"\bF\.( \d+-?\w*,?)+( \([A-Z]*\d* ?\d*\),?)*", remove_space, element)
        element = re.sub(r"\b(\d+ )?U\.S\.C\. [A-Z]+\d*", remove_space, element)
        element = re.sub(r"\b(\d+ )?U\.S\.(( |-)\(?\d+,?\)?)+", remove_space, element)
        element = re.sub(r"\bArt\. [A-Z]+,?( [A-Z]\d+)?", remove_space, element)
        element = re.sub(r"\b\d{4} [A-Z]+ \d+,?( \*\d*)?", remove_space, element)
        element = re.sub(r"\bn\.( \d+)+( \(.+?\))?", remove_space, element)
        element = re.sub(r"\bPp\. [0-9\-]+", remove_space, element)
        # element = re.sub(r"\b([A-Z]+[a-z]*[A-Z\.']+,? )+v\.( [A-Z]+[a-z]*[A-Z\.']+,?)+ ?", remove_space, element)
        element = re.sub(r"\b\d+ U\.S.[_,\- ]*\(\d+\)", remove_space, element)
        text.append(element)
    df['Text'] = text
    # Disabled as legal cases need punctuations to work
    # # remove non-alphabetical
    # df['Text'] = df['Text'].str.replace('[^a-zA-Z0-9\'\".!()]', ' ', regex=True).str.strip()


    # remove extra spaces
    df['Text'] = df['Text'].str.replace(' +', ' ', regex=True).str.strip()


    return df

def process_VS_data(df):

    #assert tokens column is present
    assert 'tokens' in df.columns

    new_tokens = []
    for case in df['tokens']:
        new_token = []
        case_found = False
        for tokenIdx in range(len(case)):
            new_token.append(case[tokenIdx])
            # Handling cases
            #United States v. Rostenkowski, 59 F. 3d 1291, 1297 (CADC 1995).
            #United States Supreme Court AARON J. SCHOCK v. UNITED STATES(2019) No. 18-406
            if case[tokenIdx] == 'v.':
                case_found = True
            elif case_found and (case[tokenIdx].startswith('No.') or case[tokenIdx][0].islower() or case[tokenIdx][0].isnumeric() or case[tokenIdx-1].lower().startswith('al.')):
                # we need to deal with this
                last_word = new_token.pop()
                castStr = ''
                while len(new_token) > 0 and (new_token[-1] == 'v.' or (new_token[-1].lower() != 'see' and new_token[-1][0].isupper())) :
                    castStr =  new_token.pop() + '|' + castStr
                new_token.append(castStr[:-1])
                case_found = False
                new_token.append(last_word)
        new_tokens.append(new_token)
    df['tokens'] = new_tokens

    # change all tokens to lower case
    df['tokens'] = df['tokens'].apply(lambda x: [item.lower() for item in x])

    return df


# In[5]:


# read data
# path = 'data/all_year.pkl.zst'
# df = read_data(path)[:15]


# ## Pre-processing here

# In[6]:

def final_preprocessing(df):
    df = preprocess(df)
    #split the processed text into tokens for further processing with pipes | as the combinator
    df['tokens'] = df['Text'].str.split() 
    df['tokens'] = process_VS_data(df)['tokens']
    df['tokens'] = tokenize_and_remove_stopwords(df)
    df['new_text'] = df['tokens'].apply(lambda x: ' '.join(x))
    #apply nlp to the new text
    df['nlp'] = df['new_text'].apply(lambda x: nlp(x))
    #for each nlp object, get lemma and pos
    df['lemma'] = df['nlp'].apply(lambda x: [token.lemma_ for token in x])
    # remove punctuations from lemma
    df['lemma'] = df['lemma'].apply(lambda x: [token for token in x if token not in string.punctuation])
    # create a new column and convert lemma to string
    df['final_text'] = df['lemma'].apply(lambda x: ' '.join(x))
    # remove '\'s' from final_text
    df['final_text'] = df['final_text'].str.replace(r'\'s', '', regex=True)
    return df


# # $\textbf{Text Summarization Functions}$

# In[7]:




def summarize_pipeline(df):
    summarizer = pipeline("summarization")
    df["summary_pipeline"] = df["final_text"].apply(lambda x: summarizer(x, truncation=True, max_length=1024, min_length=300, do_sample=False)[0]['summary_text'])
    return df


def summarize_summa(df):
    df["summary_summa"] = df["final_text"].apply(lambda x: summarizer.summarize(x, ratio=0.02))
    return df


# In[8]:


# # applied pipeline method to summarize text
# df = summarize_pipeline(df)

# # applied summa method to summarize text
# df = summarize_summa(df)


# # $\textbf{Vectorization Functions}$

# In[36]:


def concat_vectors(text_vector, transformer_vector, summa_vector):
    return np.concatenate([text_vector, transformer_vector, summa_vector],axis=0)

def combined_vectors(text_vec, transformer_vec, summa_vector, single_vec):
    
    #transform vec to numpy array
    if type(transformer_vec) != np.ndarray:
        transformer_vec = transformer_vec.toarray()
    if type(summa_vector) != np.ndarray:
        summa_vector = summa_vector.toarray()
    if type(text_vec) != np.ndarray:
        text_vec = text_vec.toarray()

    # print("CHECKKKKK ",text_vec.shape, transformer_vec.shape, summa_vector.shape, type(transformer_vec), type(summa_vector), type(text_vec))
    combined_vectors = list()
    
    # print("CHECKKKKK111111 ",text_vec.shape, transformer_vec.shape, summa_vector.shape, type(transformer_vec), type(summa_vector), type(text_vec))
    if single_vec:
        combined_vectors = concat_vectors(text_vec[0], transformer_vec[0], summa_vector[0])
    else:
        for i in range(len(transformer_vec)):
            combined_vectors.append(concat_vectors(text_vec[i], transformer_vec[i], summa_vector[i]))
    

    combined_vectors = np.array(combined_vectors)

    print("combined_vectors ", combined_vectors[0].shape, combined_vectors.shape)
    return combined_vectors


# ## Doc2Vec

# In[40]:

def doc2vec(df, single_doc):
    def tagged_document(list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
                yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    def process_doc2vec_similarity(X):

        documents = list(tagged_document(X))
            
        model = gensim.models.doc2vec.Doc2Vec(documents, vector_size=300, window=11, min_count=10, epochs=30)

        # base_vector = model.infer_vector(documents[case_id].words.split())

        vectors = []
        for i, document in enumerate(documents):

            # tokens = list(filter(lambda x: x in model.wv.vocab.keys(), document))
            vector = model.infer_vector(document.words.split())
            vectors.append(vector)

        # convert to numpy array
        vectors = np.array(vectors)

        # scores = cosine_similarity([base_vector], vectors).flatten()

        # # top 10 highest scores
        # highest_score_indices = scores.argsort()[-5:][::-1]

        
        # print(highest_score_indices)

        # highest_score = 0
        # highest_score_index = 0
        # for i, score in enumerate(scores):
        #     if highest_score < score and i != case_id:
        #         highest_score = score
        #         highest_score_index = i

        # most_similar_document = documents[highest_score_index]
        # print("Most similar document by Doc2vec with the score:", highest_score, highest_score_index)
        return vectors
    
    text_vec = process_doc2vec_similarity(df['final_text'])
    transformer_vec = process_doc2vec_similarity(df['summary_pipeline'])
    summa_vector = process_doc2vec_similarity(df['summary_summa'])

    doc2vec_vectors = combined_vectors(text_vec, transformer_vec, summa_vector, single_doc)
    return doc2vec_vectors



# In[41]:


# print("Doc2Vec")
# print("Final Text")
# text_vec = process_doc2vec_similarity(df['final_text'], 1)

# print("Summary Pipeline")
# transformer_vec = process_doc2vec_similarity(df['summary_pipeline'], 1)

# print("Summary Summa")
# summa_vector = process_doc2vec_similarity(df['summary_summa'], 1)

# print("Concatenated Vectors")
# doc2vec_vectors = combined_vectors(text_vec, transformer_vec, summa_vector)

# print(doc2vec_vectors.shape)


# ## TF-IDF

# In[34]:

def tfidf(df, single_doc):
    def vectorize_tfidf(X, num_features=1500):
        tfidf=TfidfVectorizer(min_df = 0.01, max_df=0.95, ngram_range = (1,3), max_features=num_features, norm='l2')
        #tfidf=TfidfVectorizer(ngram_range = (1,3), max_features=num_features, norm='l2')
        X_data = tfidf.fit_transform(X)
        pickle.dump(tfidf, open("tfidf_model.pkl", "wb"))
        return X_data.toarray()

    def process_tfidf_similarity(X,num_features=1500):

        if single_doc:
            transformer = pickle.load(open("tfidf_model.pkl", "rb"))
            X_data = transformer.transform(X)
        else:
            X_data = vectorize_tfidf(X,num_features)

        # cosine_similarities = linear_kernel(X_data[case_id:case_id+1], X_data).flatten()
        # related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        # print(related_docs_indices)
        return X_data
    
    text_vec = process_tfidf_similarity(df['final_text'])
    transformer_vec = process_tfidf_similarity(df['summary_pipeline'])
    summa_vector = process_tfidf_similarity(df['summary_summa'])


    tfidf_vectors = combined_vectors(text_vec, transformer_vec, summa_vector, single_doc)
    return tfidf_vectors


# In[37]:


# print("TF-IDF \nFinal Text")
# text_vec = process_tfidf_similarity(df['final_text'], 1)

# print("Summary Pipeline")
# transformer_vec = process_tfidf_similarity(df['summary_pipeline'], 1)

# print("Summary Summa")
# summa_vector = process_tfidf_similarity(df['summary_summa'], 1)


# print("Concatenated Vectors")
# tf_idf_vectors = combined_vectors(text_vec, transformer_vec, summa_vector)

# print(tf_idf_vectors.shape)


# ## BERT

# In[42]:

def bert(df, single_doc):
    def bert_similarity(X):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        sentences = X
        sentence_embeddings = model.encode(sentences)
        
        # query = sentences[case_id]
        # query_embedding = model.encode(query)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        # number_top_matches = 5
        # similar_sentences = cosine_similarity([query_embedding], sentence_embeddings)[0].argsort()[-number_top_matches:][::-1]
        # print(similar_sentences)

        return sentence_embeddings
    
    text_vec = bert_similarity(df['final_text'])
    transformer_vec = bert_similarity(df['summary_pipeline'])
    summa_vector = bert_similarity(df['summary_summa'])

    bert_vectors = combined_vectors(text_vec, transformer_vec, summa_vector, single_doc)
    return bert_vectors


# In[43]:


# print("BERT \nFinal Text")
# text_vec = bert_similarity(df['final_text'], 1)

# print("Summary Pipeline")
# transformer_vec = bert_similarity(df['summary_pipeline'], 1)

# print("Summary Summa")
# summa_vector = bert_similarity(df['summary_summa'], 1)

# print("Concatenated Vectors")
# bert_vectors = combined_vectors(text_vec, transformer_vec, summa_vector)

# print(bert_vectors.shape)


# ## Word2Vec

# In[115]:


def train_word2vec(X):
    model = Word2Vec(X, vector_size=100, window=5, min_count=1, workers=4)  
    return model



# In[132]:


# sentences = df['final_text'].apply(lambda x: x.split(" "))
# w2v_model = train_word2vec(sentences)


# # In[134]:


# # len(w2v_model.wv.index_to_key)
# w2v_model.wv.most_similar('supreme')


# In[ ]:





# In[ ]:




