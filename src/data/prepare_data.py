import enum
from typing import Dict, List

import pandas as pd
from pandas import DataFrame


class Sentiments(enum.Enum):
    POS = 'POS'
    NEG = 'NEG'


def read_sample() -> DataFrame:
    df = pd.read_csv('data/raw/reviews.csv')
    df['rating'] = df['rating'].astype(dtype='int64')

    return df


def create_classes(df: DataFrame) -> Dict[str, List[str]]:
    df['sentiment'] = df['rating'].apply(lambda x: Sentiments.POS if x>=40 else Sentiments.NEG)

    review_classes = {
        sentiment.value: df[df['sentiment'] == sentiment]['review'].values.tolist()
        for sentiment in Sentiments
    }

    return review_classes
