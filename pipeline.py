"""
Pipeline to take zst file path and document as input
and return a similar document index with its summary
"""
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.metrics.pairwise import linear_kernel
from transformers import pipeline

from pipeline_utils import *

class Pipeline:

    def __init__(self, doc_path, path_to_zst, run_type):
        self.doc_path = doc_path
        self.path_to_zst = path_to_zst
        self.run_type = run_type
        
        if self.run_type:
            print('Reading data')
            self.df = read_data(self.path_to_zst)
            self.preprocess_data()
        else:
            # read pkl file as dataframe
            self.df = pd.read_pickle('data_first15.pkl')
            self.tfidf_vec = np.load('tfidf_vec.pkl.npy')
            self.d2v_vec = np.load('d2v_vec.pkl.npy')
            self.bert_vec = np.load('bert_vec.pkl.npy')

            print("TF-IDF",self.tfidf_vec.shape, self.tfidf_vec[0].shape)
            print("Doc2Vec",self.d2v_vec.shape, self.d2v_vec[0].shape)
            print("BERT",self.bert_vec.shape, self.bert_vec[0].shape)

        self.read_doc()


    def read_doc(self):
        # read the text file and return a string
        self.doc = open(self.doc_path, 'r').read()

        self.df_target = pd.DataFrame({'Text': [self.doc]})
        self.df_target = final_preprocessing(self.df_target)
        

        self.df_target = summarize_pipeline(self.df_target)
        self.df_target = summarize_summa(self.df_target)

        self.doc_d2v = doc2vec(self.df_target, single_doc=True)
        self.doc_tfidf = tfidf(self.df_target, single_doc=True)
        self.doc_bert = bert(self.df_target, single_doc=True)
        


    def preprocess_data(self):
        print("Preprocessing data")
        self.df = final_preprocessing(self.df)

        print("Summarizing data")
        self.df = summarize_pipeline(self.df)
        self.df = summarize_summa(self.df)

        print("Doc2Vec")
        self.d2v_vec = doc2vec(self.df, single_doc=False)

        print("TF-IDF")
        self.tfidf_vec = tfidf(self.df, single_doc=False)

        print("BERT")
        self.bert_vec = bert(self.df, single_doc=False)

        print("CHECKING......")
        print("TF-IDF",self.tfidf_vec.shape, self.tfidf_vec[0].shape)
        print("Doc2Vec",self.d2v_vec.shape, self.d2v_vec[0].shape)
        print("BERT",self.bert_vec.shape, self.bert_vec[0].shape)


        np.save('d2v_vec.pkl', self.d2v_vec)
        np.save('tfidf_vec.pkl', self.tfidf_vec)
        np.save('bert_vec.pkl', self.bert_vec)


        # save df to pickle
        self.df.to_pickle('data.pkl')


    def get_similar_documents(self):

        self.doc_tfidf = self.doc_tfidf.reshape(self.doc_tfidf.shape[0], 1).T
        self.doc_d2v = self.doc_d2v.reshape(self.doc_d2v.shape[0], 1).T
        self.doc_bert = self.doc_bert.reshape(self.doc_bert.shape[0], 1).T

 
        print("TF-IDF SIMILARITY")
        cosine_similarities_tfidf = cosine_similarity(self.doc_tfidf, self.tfidf_vec).flatten()
        related_docs_tfidf = cosine_similarities_tfidf.argsort()[-5:][::-1]
        print(related_docs_tfidf)


        print("Doc2Vec SIMILARITY")
        cosine_similarities_d2v = cosine_similarity(self.doc_d2v, self.d2v_vec).flatten()
        related_docs_d2v = cosine_similarities_d2v.argsort()[-5:][::-1]
        print(related_docs_d2v)


        print("BERT SIMILARITY")
        cosine_similarities_bert = cosine_similarity(self.doc_bert, self.bert_vec)[0]
        related_docs_bert = cosine_similarities_bert.argsort()[-5:][::-1]
        print(related_docs_bert)


if __name__ == '__main__':
    pipeline_obj = Pipeline('check1.txt', 'data/all_year.pkl.zst', run_type=True)
    pipeline_obj.get_similar_documents()
    
            

        

"""
TF-IDF SIMILARITY
[ 0  8 12  5 14]
Doc2Vec SIMILARITY
[11 10 12 13  5]
BERT SIMILARITY
[ 0 14  1  5  6]
"""

"""
TF-IDF SIMILARITY
[ 0  8  5 14 12]
Doc2Vec SIMILARITY
[11 10 13 12  9]
BERT SIMILARITY
[ 0  6  5 14  2]
"""


