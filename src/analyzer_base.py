import os
import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

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
        self.all_instance_list = []

        self.full_data_path = str()
        self.cleaned = False
        self.cleaning_kwargs = {}

        self._load_data()

        # TODO add Naive Bayes
        self.model_choices = [LogisticRegression, RandomForestRegressor]

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
    def evaluate_model(self, cv=5, scoring="accuracy"):
        overfit = False
        # evaluate
        self.model.score(self.X_train, self.y_train), self.model.score(
            self.X_test, self.y_test)
        # If the training score >> test score then the model is overfitting
        # if training score << mean(validation_scores), you might be underfitting!
        training_score = self.model.score(self.X_train, self.y_train)
        test_score = round(self.model.score(self.X_test, self.y_test), 2)
        print(f'train: {training_score}, test: { test_score}')
        # 1% not such a big deal
        if (training_score * 0.95) > test_score:
            print("On no, Overfit!")
            overfit = True
        else:
            if training_score > test_score:
                print("You're close to overfitting but within 5% so we'll let it pass")
        cross_log = cross_val_score(self.model,        # estimator: model that you want to evaluate
                                    self.X_train,  # training input data
                                    self.y_train,  # training output data
                                    cv=cv,     # number of validation datasets, k-folds
                                    scoring=scoring)  # evaluation metric
        try:
            name = f"{self.model.__name__} with {str(self.model_kwargs)}"
        except AttributeError:
            name = f"{self.model.__class__.__name__} with {str(self.model_kwargs)}"
        evaluation_dict = {"Name": name, "Cross Log Mean Score": cross_log.mean(), "Overfit": overfit,
                           "Training Score": training_score, "Test Score": test_score
                           }
        self.all_instance_list.append(evaluation_dict)

    @timeit
    def perform_grid_search(self, grid, n_splits=10, n_repeats=3, random_state=1, scoring='neg_mean_absolute_error', n_jobs=-1):

        cv = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats, random_state=random_state)
        # define search
        search = GridSearchCV(
            self.model, grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
        # perform the search
        results = search.fit(self.X_train, self.y_train)
        # summarize
        print('Best Score: %.3f' % results.best_score_)
        print('Best Config: %s' % results.best_params_)
        return results

    @timeit
    def run_model(self, model_class, **kwargs):
        if model_class not in self.model_choices:
            print(f"Model {model_class.__name__} not recognized")
            return

        self.model_kwargs = dict(kwargs)
        if self.verbose:
            print(f"Starting run_model wih {model_class.__name__}")

        # NOTE:
        # Logistic Regression, Perceptron, Support Vector Machines, are binary
        # So they need OneVsRestClassifier for Kaggle data bvc it has 6 classes
        if model_class.__name__ == "LogisticRegression":
            log_reg = LogisticRegression(**kwargs)
            self.model = OneVsRestClassifier(log_reg)
        else:
            self.model = model_class(**kwargs)

        self.model.fit(self.X_train, self.y_train)

    def build_results_df(self):
        results_data = [(i['name'], i['cross_log_mean'], i['is_overfit'],  i['training_score'],
                         i['test_score']) for i in self.all_instance_list]
        return pd.DataFrame(results_data, columns=["Name", "Cross Log Mean Score", "Overfit", "Training Score", "Test Score"])

    def make_pred_array(self, test_str):
        """
        input is a string to test , like "I am just a wanderlust king gypsy punk boy ai"
        output is array of word count dictionaries
        """
        def word_count(test_str):
            counts = dict()
            words = test_str.split()

            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1

            return counts

        word_counts = word_count(test_str)
        word_array = []
        for c in self.X.columns:
            count = word_counts.get(c)
            if count:
                word_array.append(count)
            else:
                word_array.append(0)

        return word_array

    def get_toxicoty_scores(self, text_dict_list):
        """
        text_dict_list is a list of dictionries with {id:text}
        """
        for td in text_dict_list:
            text = self.processor.normalize_corpus([td["text"]])
            pred_array = self.make_pred_array(text)
            score = self.model.predict_proba([pred_array]).round(4)

            # modified_score = int(score * 10)  # is 10 too harsh?
            #modified_score = int(score * 5)
            #td["score"] = modified_score
            td["score"] = score
            # I want score to be an int between 0-10
            # 5 is already unreadable
        return text_dict_list
