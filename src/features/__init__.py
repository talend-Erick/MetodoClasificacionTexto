import spacy


nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])