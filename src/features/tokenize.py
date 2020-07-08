import pickle
from typing import Dict, List

from src.features import nlp
from src.features.utils import (sentences_to_words, remove_stopwords,
                                bigrams_model, extend_bigrams, lemmatization)


def tokenize_classes(document_classes: Dict[str, List[str]], load_bigrams: bool = True) -> Dict[str, List[List[str]]]:

    word_classes = {}
    for document_class, documents in document_classes.items():
        word_classes[document_class] = list(sentences_to_words(documents))
        word_classes[document_class] = remove_stopwords(word_classes[document_class])

    if load_bigrams:
        with open('models/bigrams.pkl', 'rb') as input_file:
            bigram_mod = pickle.load(input_file)
    else:
        words = []
        for word_class, words in word_classes.items():
            words.extend(words)
        bigram_mod = bigrams_model(words)
        with open('models/bigrams.pkl', 'wb') as output_file:
            pickle.dump(bigram_mod, output_file)

    for word_class, words in word_classes.items():
        word_classes[word_class] = extend_bigrams(words, bigram_mod)
        word_classes[word_class] = lemmatization(nlp, word_classes[word_class])

    return word_classes
