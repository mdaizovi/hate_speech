import contractions  # https://github.com/kootenpv/contractions
# from pycontractions import Contractions # recomended over contractions but has installation problem
# https://pypi.org/project/pycontractions/
import textstat  # https://pypi.org/project/textstat/
# https://pypi.org/project/gibberish-detector/
from gibberish_detector import detector
# https://github.com/snguyenthanh/better_profanity
from better_profanity import profanity
import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

import spacy
# You can now load the package via spacy.load('en_core_web_sm')

# nltk.download('stopwords')
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

#nlp = spacy.load('en_core', parse=True, tag=True, entity=True)
nlp = spacy.load('en_core_web_sm')

"""
Handles the cleaning of raw text before feeding through model,
as well as presentation of text afterwards (ie censoring offensive words)
"""

tokenizer = ToktokTokenizer()


class TextPreprocessor:

    def __init__(self):
        pass

    def expand_contractions(self, s, slang=True):
        """
        Accept string, replace contractions in strong with words.
        ex:
        INPUT: ive gotta go! i'll see yall later.
        OUTPUT: I have got to go! I will see you all later.
        """
        return contractions.fix(s, slang=slang)

    def remove_special_characters(self, s, remove_digits=False):
        """
        Removes all non-letters from text.
        NOTE if you use this, use it AFTER replacing leetspeak
        """
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        return re.sub(pattern, '', s)

    def replace_accented_chars(self, s):
        """
        Replaces accented chars with ascii.
        ex:
        INPUT: Sómě Áccěntěd těxt
        OUTPUT: Some Accented text
        """
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def simple_stemmer(self, s):
        """
        Obtaining the base form of a word from its inflected form is known as stemming. 
        Stemming helps us in standardizing words to their base or root stem, 
        irrespective of their inflections, which helps many applications like classifying 
        or clustering text, and even in information retrieval.
        ex:
        INPUT: My system keeps crashing his crashed yesterday, ours crashes dail
        OUTPUT: My system keep crash hi crash yesterday, our crash daili
        """

        ps = nltk.porter.PorterStemmer()
        return ' '.join([ps.stem(word) for word in s.split()])

    def lemmatize_text(self, s):
        """
        Lemmatization is very similar to stemming, where we remove word affixes to get to the
        base form of a word. However, the base form in this case is known as the root word, 
        but not the root stem. The difference being that the root word is always a 
        lexicographically correct word (present in the dictionary), but the root stem may not be so. 
        ex:
        INPUT: My system keeps crashing! his crashed yesterday, ours crashes daily
        OUTPUT: My system keep crash ! his crash yesterday , ours crash daily
        """
        text = nlp(s)
        return ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    def remove_stopwords(self, s, is_lower_case=False):
        """
        Words which have little or no significance, 
        especially when constructing meaningful features from text, are known as stopwords or stop words. 
        These are usually words that end up having the maximum 
        frequency if you do a simple term or word frequency in a corpus.
        ex:
        INPUT: The, and, if are stopwords, computer is not
        OUTPUT: , , stopwords , computer not
        """
        tokens = tokenizer.tokenize(s)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [
                token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [
                token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def normalize_corpus(self, input_corpus, html_stripping=True, contraction_expansion=True,
                         accented_char_removal=True, text_lower_case=True,
                         stem_text=False, text_lemmatization=True,
                         special_char_removal=True,
                         stopword_removal=True, remove_digits=True):

        # if corpus is a str, return str
        # if a list of strs, return list of strs
        if isinstance(input_corpus, str):
            corpus = [input_corpus]
        elif isinstance(input_corpus, list):
            corpus = input_corpus

        normalized_corpus = []
        # normalize each document in the corpus
        for doc in corpus:

            # Standard tet cleaning
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
            # lowercase the text
            if text_lower_case:
                doc = doc.lower()
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)

            # More specialized text cleaning, not all methoa may be appriopriate
            # remove accented characters
            if accented_char_removal:
                doc = self.replace_accented_chars(doc)
            # expand contractions
            if contraction_expansion:
                doc = self.expand_contractions(doc)
            # NOTE you will prob stem OR lemmatize but not both
            # stem text
            if stem_text:
                doc = self.simple_stemmer(doc)
            # lemmatize text
            if text_lemmatization:
                doc = self.lemmatize_text(doc)
            # remove special characters and\or digits
            if special_char_removal:
                # insert spaces between special characters to isolate them
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special_characters(
                    doc, remove_digits=remove_digits)
            # remove stopwords
            if stopword_removal:
                doc = self.remove_stopwords(doc, is_lower_case=text_lower_case)

            normalized_corpus.append(doc)

        # if corpus is a str, return str
        # if a list of strs, return list of strs
        if isinstance(input_corpus, str):
            output_corpus = normalized_corpus[0]
        elif isinstance(input_corpus, list):
            output_corpus = normalized_corpus

        return output_corpus

    def string_has_profanity(self, s):
        """
        Accept string s.
        If profanity is in string, including in leetspeak, return True 
        """
        return profanity.contains_profanity(s)

    def replace_profanity(self, s, replacement_char="-"):
        """
        Accept string s.
        if profanity is found, replaces it with selected char.
        Used for presentation, not for model training.
        EX
        INPUT: You p1ec3 of sHit.
        OUTPUT: You ---- of ----.
        """
        return profanity.censor(self, s, replacement_char)
