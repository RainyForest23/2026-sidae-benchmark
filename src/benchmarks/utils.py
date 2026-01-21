
import random
from datasets import Dataset, DatasetDict

def sample_dataset(dataset, sample_size=None, seed=42):
    """
    Samples from a dataset or dataset dict.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
        sample_size: int (number of samples) or float (fraction) or None (no sampling)
        seed: random seed
        
    Returns:
        Sampled dataset
    """
    if sample_size is None:
        return dataset

    if isinstance(dataset, DatasetDict):
        sampled_dict = DatasetDict()
        for split in dataset.keys():
            sampled_dict[split] = sample_dataset(dataset[split], sample_size, seed)
        return sampled_dict

    # It's a Dataset
    total_rows = len(dataset)
    if isinstance(sample_size, float):
        n = int(total_rows * sample_size)
    else:
        n = min(sample_size, total_rows)
        
    if n >= total_rows:
        return dataset
        
    indices = list(range(total_rows))
    random.seed(seed)
    random.shuffle(indices)
    sampled_indices = indices[:n]
    
    return dataset.select(sampled_indices)
