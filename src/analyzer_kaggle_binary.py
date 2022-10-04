import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from settings import DATA_DIR
from analyzer_base import AnalyzerBaseClass
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Uses Kaggle dataset but combines 6 columns to 1.
Columns: toxic, sever_toxic, obscene, threat, insult, identity_hate
"""


class KaggleBinaryAnalyzer(AnalyzerBaseClass):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_field = 'comment_text'

    def _prepare_data(self):
        print("preparing data doing something")
        all_hate_speech_columns = ['toxic', 'severe_toxic',
                                   'obscene', 'threat', 'insult', 'identity_hate']
        # Actually I don't think toxic, obscene, threat, or insult need to be included.
        blockable_hate_speech_columns = ['severe_toxic', 'identity_hate']
        # combine hate_speech_columns so if any 1 is true we'll get a 1 for the new binary "deplorable" coulmn
        self.train["blockable"] = self.train[blockable_hate_speech_columns].max(
            axis=1)
        self.train = self.train.drop(all_hate_speech_columns, axis=1)

    def tt_split(self, test_size=0.2, random_state=1):

        self.X = self.train.comment_text.values.astype('U')
        self.y = self.train.drop(['id', 'comment_text'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        for k, v in {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}.items():
            setattr(self, k, v)
        if self.verbose:
            print("Train and Test split")

    def vectorize(self):
        # The process to convert text data into numerical data/vector,
        # is called vectorization or in the NLP world, word embedding.
        # Bag-of-Words(BoW) and Word Embedding (with Word2Vec) are
        # two well-known methods for converting text data to numerical data.

        # One of the major drawbacks of using Bag-of-words techniques is that
        # it can’t capture the meaning or relation of the words from vectors.
        # Word2Vec is one of the most popular technique to learn word embeddings
        # using shallow neural network which is capable of capturing context of a
        # word in a document, semantic and syntactic similarity, relation with other words, etc.
        self.tfidf_vectorizer = TfidfVectorizer(
            use_idf=True, analyzer='word', stop_words='english')
        self.X_train = self.tfidf_vectorizer.fit_transform(self.X_train)
        self.X_test = self.tfidf_vectorizer.transform(self.X_test)

    def predict_unlabeled_df(self, df, **kwargs):
        """
        Input is a df of columns [id, comment_text] to be scored
        output is df of [id, comment_text, score] to be used in extension
        """

        # use same cleaning_kwargs as data was cleaned with
        df[self.text_field] = df[self.text_field].apply(
            lambda x: self.processor.normalize_corpus(input_corpus=x, **kwargs))
        X_test = df[self.text_field]

        X_vector = self.tfidf_vectorizer.transform(X_test)
        y_prob = self.model.predict_proba(X_vector)[:, 1]
        df['score'] = y_prob
        y_predict = self.model.predict(X_vector)
        df['blockable'] = y_predict # unnecessary ? 
        df.set_index('id', inplace=True)
        return df


#from analyzer_kaggle_binary import*
# ka = KaggleBinaryAnalyzer(train_filepath="kaggle_binary/train.csv",
#                     test_filepath="kaggle_binary/test.csv")
# ka.tt_split()
# ka.vectorize()

# ka.run_model(model_class=RandomForestRegressor)
# or
# ka.run_model(model_class=LogisticRegression)
# ka.evaluate_model(cv=5, scoring="accuracy") # just plain LR gets me
#   # train: 0.9864636209813875, test: 0.99

# test = ka.predict_unlabeled_df(ka.test_df)
# test.head(12)