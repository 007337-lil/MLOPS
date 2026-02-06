import pandas as pd
import numpy as np

def normalize_columns(columns):
    new_cols = []

    for col in columns:
        col = col.lower()
        col = re.sub(r"[^a-z0-9]+", " ", col)
        words = col.split()
        short_words = [w[:3] for w in words]
        new_cols.append("_".join(short_words))

    return new_cols