import pickle
from typing import Generator, List

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from spacy.lang.en import English


def sentences_to_words(sentences: List[str]) -> Generator:
    for sentence in sentences:
        # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True elimina la puntuación


def remove_stopwords(documents: List[List[str]]) -> List[List[str]]:
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords.words('english')]
            for doc in documents]


def bigrams_model(documents: List[List[str]], save: bool = False) -> Phraser:
    # We learn bigrams
    #  https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
    bigram = Phrases(documents, min_count=5, threshold=10)

    # we reduce the bigram model to its minimal functionality
    bigram_mod = Phraser(bigram)

    if save:
        with open('../../models/bigrams.pkl', 'wb') as output_file:
            pickle.dump(bigram_mod, output_file)

    return bigram_mod


def extend_bigrams(documents: List[List[str]], bigram_mod: Phraser) -> List[List[str]]:
    # we apply the bigram model to our documents
    return [bigram_mod[doc] for doc in documents]


def lemmatization(nlp: English, texts: List[List[str]], allowed_postags: List = None) -> List[List[str]]:
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out