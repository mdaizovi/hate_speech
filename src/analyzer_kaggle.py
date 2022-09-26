import os
import pandas as pd

from sklearn.model_selection import train_test_split

"""
Base Class for analyzing tezt.
Analyzers for specific datasets will inherit and subclass as necessary. 
"""

from settings import DATA_DIR
from analyzer_base import AnalyzerBaseClass


class KaggleAnalyzer(AnalyzerBaseClass):
    # test columns: ["id", "comment_text"]
    # train columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_field = 'comment_text'

    def prepare_data(self):
        pass

    def tt_split(self, test_size=0.2, random_state=1):
        X = self.train.comment_text
        y = self.train.drop(['id', 'comment_text'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        for k, v in {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}.items():
            setattr(self, k, v)
        if self.verbose:
            print("Train and Test split")


ka = KaggleAnalyzer(train_filepath="kaggle/train.csv",
                    test_filepath="kaggle/test.csv")
ka.tt_split()
ka.clean_data()
ka.output_cleaned_data()
# ka.do_logistic_regression(
#     C=10, penalty='l2', solver='liblinear', random_state=4)
