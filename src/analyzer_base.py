import os
import pandas as pd

"""
Base Class for analyzing tezt.
Analyzers for specific datasets will inherit and subclass as necessary. 
"""

from settings import DATA_DIR
from text_processor import TextPreprocessor


class AnalyzerBaseClass:

    def __init__(self, train_filepath, test_filepath):
        self.processor = TextPreprocessor()
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.text_field = str()

        self._load_data()

    def _load_data(self, encoding="ISO-8859-1"):
        train_full_path = os.path.join(DATA_DIR, self.train_filepath)
        test_full_path = os.path.join(DATA_DIR, self.test_filepath)

        self.train = pd.read_csv(train_full_path, encoding=encoding)
        self.test = pd.read_csv(test_full_path, encoding=encoding)

    def prepare_data(self):
        """
        Preparation specific to dataset, if needed. Dropping columns, removing HTML, etc.
        """
        raise NotImplementedError()

    def clean_data(self):
        self.train[self.text_field] = self.train[self.text_field].apply(
            lambda x: self.processor.normalize_corpus([x]))
