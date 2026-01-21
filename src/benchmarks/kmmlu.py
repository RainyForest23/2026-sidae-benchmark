
from datasets import load_dataset
from .utils import sample_dataset

# Full list of KMMLU categories
CATEGORIES = [
    'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 
    'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 
    'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics', 
    'Education', 'Electrical-Engineering', 'Electronics-Engineering', 
    'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing', 
    'Gas-Technology', 'Geomatics-and-Land-Surveying', 'Health-Care-Management', 
    'History', 'Human-Resources-Management', 'Industrial-Engineer', 
    'Information-Technology', 'Interior-Architecture-Design', 'Korean-History', 
    'Law', 'Machine-Design', 'Management', 'Maritime-Engineering', 'Marketing', 
    'Materials-Engineering', 'Math', 'Mechanical-Engineering', 'Nondestructive-Testing', 
    'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety', 
    'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 
    'Social-Welfare', 'Taxation', 'Telecommunications-and-Wireless-Technology', 
    'TM-Chemical', 'TM-Mechanical', 'TM-Metallurgical'
]
# Note: The above list is approximate/common. 
# Based on the inspection, we can start with 'Accounting' and maybe discover others or use "all".
# HuggingFace usually allows loading "all" or we iterate. 
# For KMMLU, typically we load specific configs.

def load_kmmlu(category, sample_size=None, seed=42):
    ds = load_dataset("HAERAE-HUB/KMMLU", category)
    return sample_dataset(ds, sample_size, seed)
