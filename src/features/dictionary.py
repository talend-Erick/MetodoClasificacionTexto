from typing import List, Tuple

from gensim.corpora import Dictionary


def create_dictionary(documents: List[List[str]]):
    return Dictionary(documents)


def term_document_matrix(documents: List[List[str]], dictionary: Dictionary) -> List[List[Tuple[int, int]]]:
    return [dictionary.doc2bow(text) for text in documents]
