
from datasets import load_dataset
from .utils import sample_dataset

TASKS = ['boolq', 'copa', 'hellaswag', 'sentineg', 'wic']

def load_kobest(task, sample_size=None, seed=42):
    if task not in TASKS:
        raise ValueError(f"KoBEST task must be one of {TASKS}")
    
    ds = load_dataset("skt/kobest_v1", task)
    return sample_dataset(ds, sample_size, seed)
