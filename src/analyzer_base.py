import os
import pandas as pd
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score, confusion_matrix, f1_score
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

        self.full_data_path = str()
        self.cleaned = False
        self.cleaning_kwargs = {}

        self._load_data()

    def _load_data(self, encoding="ISO-8859-1"):

        self.full_data_path = os.path.join(
            DATA_DIR, pathlib.Path(self.train_filepath).parent)

        # if cleaned csv exists, use that:
        cleaned_filename = "train_cleaned.csv"
        full_cleaned_filepath = os.path.join(
            self.full_data_path, cleaned_filename)
        full_cleaned_file = pathlib.Path(full_cleaned_filepath)
        if full_cleaned_file.is_file():
            self.cleaned = True
            active_train_filepath = full_cleaned_filepath
        else:
            train_full_path = os.path.join(DATA_DIR, self.train_filepath)
            active_train_filepath = train_full_path
        self.train = pd.read_csv(active_train_filepath, encoding=encoding)
        if self.verbose:
            print(f"using file {active_train_filepath}")

        test_full_path = os.path.join(DATA_DIR, self.test_filepath)
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
            lambda x: self.processor.normalize_corpus(input_corpus=x, **kwargs))
        self.cleaned = True
        if self.verbose:
            print("Data is clean")

    def output_cleaned_train_data(self):
        if not self.cleaned:
            self.clean_data()

        cleaned_full_path = os.path.join(
            self.full_data_path, 'train_cleaned.csv')
        self.train.to_csv(cleaned_full_path, index=False)

    @timeit
    def do_logistic_regression(self, **kwargs):
        if self.verbose:
            print("Starting Logistic Regression")
        log_reg = LogisticRegression(**kwargs)
        self.model = OneVsRestClassifier(log_reg)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, scoring="accuracy"):
        # evaluate
        self.model.score(self.X_train, self.y_train), self.model.score(
            self.X_test, self.y_test)
        # How can we check how well the model performed? Calculate the accuracy
        # If the training score >> test score then the model is overfitting
        # if training score << mean(validation_scores), you might be underfitting!
        training_score = self.model.score(self.X_train, self.y_train)
        test_score = round(self.model.score(self.X_test, self.y_test), 2)
        print(f'train: {training_score}, test: { test_score}')
        # 1% not such a big deal
        if (training_score * 0.95) > test_score:
            print("On no, Overfit!")
        else:
            if training_score > test_score:
                print("You're close to overfitting but within 5% so we'll let it pass")

        y_train_pred_proba = self.model.predict_proba(self.X_train)
        y_valid_pred_proba = self.model.predict_proba(self.X_test)

        roc_auc_score_train = roc_auc_score(
            self.y_train, y_train_pred_proba, average='weighted')
        roc_auc_score_test = roc_auc_score(
            self.y_test, y_valid_pred_proba, average='weighted')

        print("ROC AUC Score Train:", roc_auc_score_train)
        print("ROC AUC Score Test:", roc_auc_score_test)
        prediction = self.model.predict(self.X_test)
        print('Accuracy Score: ', accuracy_score(self.y_test, prediction))
        print('hamming loss : ', hamming_loss(self.y_test, prediction))

    @timeit
    def do_naive_bayes(self, **kwargs):
        if self.verbose:
            print("Starting Naive Bayes")

    @timeit
    def do_random_forest(self, **kwargs):
        if self.verbose:
            print("Starting Random Forest")

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
