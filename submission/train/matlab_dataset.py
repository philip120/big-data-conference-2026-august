import torch
from torch.utils.data import Dataset
import signal
from train.load_dataset import load_matlab_nl_dataset
from shared.semantic_extractor import SemanticExtractor

class MatlabPseudocodeDataset(Dataset):
    """
    Dataset wrapper for Hugging Face philip120/matlab-nl-pseudocode
    Pre-computes semantic features to avoid parsing overhead during training.
    """

    def __init__(self, split: str = "train", model_type: str = "vit"):
        print(f"Loading dataset from Hugging Face (split={split})...")
        raw_data = load_matlab_nl_dataset(split)

        self.data = []
        if model_type in ("combined", "tree", "tree_text"):
            from model2.semantic_extractor import SemanticExtractorV2
            extractor = SemanticExtractorV2()
        else:
            extractor = SemanticExtractor()
        
        print(f"Preprocessing {len(raw_data)} samples (extracting semantic features)...")
        # Use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(raw_data)
        except ImportError:
            iterator = raw_data
        
        # Timeout handler for parsing
        def handler(signum, frame):
            raise TimeoutError("Timeout")

        # Register handler only on Unix-like systems
        use_timeout = hasattr(signal, 'SIGALRM')
        if use_timeout:
            signal.signal(signal.SIGALRM, handler)

        for sample in iterator:
            try:
                # Set 2 second timeout for parsing
                if use_timeout:
                    signal.alarm(2)
                
                features = extractor(sample['code'])
                
                if use_timeout:
                    signal.alarm(0)

                if not features['texts']:
                    continue
                
                self.data.append({
                    'code': sample['code'],
                    'target': sample['nl'],
                    'features': features
                })
            except Exception:
                if use_timeout:
                    signal.alarm(0)
                continue
                
        print(f"Loaded {len(self.data)} valid samples (filtered {len(raw_data) - len(self.data)} bad samples)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
