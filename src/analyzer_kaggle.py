import os
import pandas as pd

"""
Base Class for analyzing tezt.
Analyzers for specific datasets will inherit and subclass as necessary. 
"""

from settings import DATA_DIR
from analyzer_base import AnalyzerBaseClass


class KaggleAnalyzer(AnalyzerBaseClass):
    # test columns: ["id", "comment_text"]
    # train columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

    def __init__(self):
        super(self).__init__()
        self.text_field = 'comment_text'

    def prepare_data(self):
        pass


ka = KaggleAnalyzer(train_filepath="kaggle/train.csv",
                    test_filepath="kaggle/test.csv")
