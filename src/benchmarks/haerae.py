
from datasets import load_dataset
from .utils import sample_dataset

TASKS = [
    'correct_definition_matching', 'csat_geo', 'csat_law', 'csat_socio', 
    'date_understanding', 'general_knowledge', 'history', 'loan_words', 
    'lyrics_denoising', 'proverbs_denoising', 'rare_words', 
    'standard_nomenclature', 'reading_comprehension'
]

def load_haerae(task, sample_size=None, seed=42):
    if task not in TASKS:
        raise ValueError(f"HAE-RAE task must be one of {TASKS}")
    
    ds = load_dataset("HAERAE-HUB/HAE_RAE_BENCH_1.1", task)
    return sample_dataset(ds, sample_size, seed)
