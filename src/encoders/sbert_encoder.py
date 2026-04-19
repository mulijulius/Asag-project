import numpy as np
from sentence_transformers import SentenceTransformer
from joblib import Memory
import os

class SBERTEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir=None):
        self.model = SentenceTransformer(model_name)
        self.memory = Memory(cache_dir or os.path.join(os.getcwd(), 'joblib_cache'), verbose=0)

    @self.memory.cache
    def encode(self, texts):
        return np.array(self.model.encode(texts, convert_to_numpy=True))

    @self.memory.cache
    def encode_pairs(self, text_pairs):
        return np.array(self.model.encode(text_pairs, convert_to_numpy=True, show_progress_bar=True))