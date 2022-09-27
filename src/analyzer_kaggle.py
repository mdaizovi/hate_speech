import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from settings import DATA_DIR
from analyzer_base import AnalyzerBaseClass
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Base Class for analyzing tezt.
Analyzers for specific datasets will inherit and subclass as necessary. 
"""


class KaggleAnalyzer(AnalyzerBaseClass):
    # test columns: ["id", "comment_text"]
    # train columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_field = 'comment_text'

    def prepare_data(self):
        pass

    def tt_split(self, test_size=0.2, random_state=1):
        # TODO Do i need TfidfVectorizer for all models or just logisticRegression?
        vectorizer = TfidfVectorizer(
            analyzer='word', stop_words='english', ngram_range=(1, 3))
        X = vectorizer.fit_transform(
            self.train.comment_text.values.astype('U'))
        y = self.train.drop(['id', 'comment_text'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        for k, v in {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}.items():
            setattr(self, k, v)
        if self.verbose:
            print("Train and Test split")


# from analyzer_kaggle import*


ka = KaggleAnalyzer(train_filepath="kaggle/train.csv",
                    test_filepath="kaggle/test.csv")
ka.tt_split()

# ka.run_model(model_class=LogisticRegression)
# ka.evaluate_model()
# I prob don't need to do this twice
# ka.run_model(model_class=LogisticRegression, C=10,
#              penalty='l2', solver='liblinear', random_state=4)
# ka.evaluate_model()

ka.run_model(model_class=RandomForestRegressor)
ka.evaluate_model()

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
}
grid_search_results = ka.perform_grid_search(grid=random_grid)


results_df = ka.build_results_df()
results_df.sort_values(by=["overfit", 'training_score',
                       'Cross Log Mean Score'], ascending=[True, False, False])
results_df.head(5)
