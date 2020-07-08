import math
import random
from typing import Dict, List, Tuple


def split_data(data: List[str], weights: Tuple = (0.8, 0.2, 0.0), seed: int = 100) -> Dict:
    split = {}

    random.seed(seed)
    random.shuffle(data)

    total_words = len(data)
    train_limit = math.floor(total_words * weights[0])
    test_limit = math.floor(total_words * weights[1] + total_words * weights[0])

    split['train'] = data[:train_limit]
    split['test'] = data[train_limit:test_limit]
    split['validation'] = data[test_limit:]

    return split
