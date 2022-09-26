import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier

from settings import DATA_DIR
from text_processor import TextPreprocessor
from utils import timeit

"""
Base Class for analyzing tezt.
Analyzers for specific datasets will inherit and subclass as necessary. 
"""


class AnalyzerBaseClass:

    def __init__(self, train_filepath, test_filepath, verbose=True):
        self.verbose = verbose
        self.processor = TextPreprocessor()
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.text_field = str()

        self.cleaned = False
        self.cleaning_kwargs = {}

        self._load_data()

    def _load_data(self, encoding="ISO-8859-1"):

        # cleaned.csv

        train_full_path = os.path.join(DATA_DIR, self.train_filepath)
        test_full_path = os.path.join(DATA_DIR, self.test_filepath)

        self.train = pd.read_csv(train_full_path, encoding=encoding)
        self.test = pd.read_csv(test_full_path, encoding=encoding)
        if self.verbose:
            print("\nData loaded")

    def prepare_data(self):
        """
        Preparation specific to dataset, if needed. Dropping columns, removing HTML, etc.
        """
        raise NotImplementedError()

    def tt_split(self):
        """
        split X,y, and perform train_test_split 
        """
        raise NotImplementedError()

    @timeit
    def clean_data(self, **kwargs):
        self.cleaning_kwargs = dict(kwargs)  # is empty if defaults are used
        if self.verbose:
            print(f"Starting to clean data with kwargs {self.cleaning_kwargs}")
        self.train[self.text_field] = self.train[self.text_field].apply(
            lambda x: self.processor.normalize_corpus(corpus=[x], **kwargs))
        self.cleaned = True
        if self.verbose:
            print("Data is clean")

    def output_cleaned_data(self):
        if not self.cleaned:
            self.clean_data()
        self.train.to_csv('cleaned.csv', index=False)

    @timeit
    def do_logistic_regression(self, **kwargs):
        if self.verbose:
            print("Starting Logistic Regression")
        # log_reg = LogisticRegression(C = 10, penalty='l2', solver = 'liblinear', random_state=45)
        log_reg = LogisticRegression(**kwargs)
        self.model = OneVsRestClassifier(log_reg)
        self.model.fit(self.X_train, self.y_train)

        y_train_pred_proba = self.model.predict_proba(self.X_train)
        y_valid_pred_proba = self.model.predict_proba(self.X_test)

        roc_auc_score_train = roc_auc_score(
            self.y_train, y_train_pred_proba, average='weighted')
        roc_auc_score_test = roc_auc_score(
            self.y_true, y_valid_pred_proba, average='weighted')

        if self.verbose:
            print("ROC AUC Score Train:", roc_auc_score_train)
            print("ROC AUC Score Test:", roc_auc_score_test)

    def text_is_toxic(self, text, model):
        # Don't know if this works
        text = self.processor.normalize_corpus([text])
        y_test_pred = model.predict_proba(text)
        result = sum(y_test_pred[0])
        if result >= 1:
            if self.verbose:
                print(f"Toxic: {result}")
            return True
        else:
            if self.verbose:
                print(f"Not toxic: {result}")
            return False
