"""
Pipeline to take zst file path and document as input
and return a similar document index with its summary
"""
import pandas as pd
import numpy as np
import re
import nltk

from pipeline_utils import *

class Pipeline:

    def __init__(self, doc_path, path_to_zst, run_type):
        self.doc_path = doc_path
        self.path_to_zst = path_to_zst
        self.run_type = run_type
        
        if self.run_type:
            print('Reading data')
            self.df = read_data(self.path_to_zst)[:15]
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
        self.df = final_preprocessing(self.df)

        self.df = summarize_pipeline(self.df)
        self.df = summarize_summa(self.df)

        self.d2v_vec = doc2vec(self.df, single_doc=False)
        self.tfidf_vec = tfidf(self.df, single_doc=False)
        self.bert_vec = bert(self.df, single_doc=False)

        print("CHECKING.....")
        print("TF-IDF",self.tfidf_vec.shape, self.tfidf_vec[0].shape)
        print("Doc2Vec",self.d2v_vec.shape, self.d2v_vec[0].shape)
        print("BERT",self.bert_vec.shape, self.bert_vec[0].shape)


        np.save('d2v_vec.pkl', self.d2v_vec)
        np.save('tfidf_vec.pkl', self.tfidf_vec)
        np.save('bert_vec.pkl', self.bert_vec)


        # save df to pickle
        self.df.to_pickle('data_first15.pkl')


    def get_similar_documents(self):

        # get cosine similarity with tfidf
        print(self.doc_tfidf.shape, self.tfidf_vec.shape)
        print(self.doc_d2v.shape, self.d2v_vec.shape)
        print(self.doc_bert.shape, self.bert_vec.shape)


        print('#'*50)
        cosine_similarities_tfidf = cosine_similarity(self.doc_tfidf, self.tfidf_vec).flatten()

        #cosine_similarities = linear_kernel(X_data[case_id:case_id+1], X_data).flatten()
        related_docs_indices = cosine_similarities_tfidf.argsort()[:-5:-1]
        print(related_docs_indices)


pipeline_obj = Pipeline('check.txt', 'data/all_year.pkl.zst', run_type=True)
pipeline_obj.get_similar_documents()
        

    

    


