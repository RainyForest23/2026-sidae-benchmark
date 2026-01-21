
from datasets import load_dataset
from .utils import sample_dataset

def load_logickor(sample_size=None, seed=42):
    ds = load_dataset("maywell/LogicKor")
    return sample_dataset(ds, sample_size, seed)
