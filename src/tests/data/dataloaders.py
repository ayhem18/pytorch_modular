"""
This script tests whether the dataloaders are reproducible across different runs (after multiple declarations + definitions) without returning the exact same sequence of indices 
"""

from torch.utils.data.dataset import Dataset
from mypt.data.dataloaders.standard_dataloaders import initialize_train_dataloader

class _IntegerDataset(Dataset):
    def __init__(self, length:int) -> None:
        super().__init__()
        self.length = length

    def __len__(self):
        return self.length
    
    def __getitem__(self, index) -> int:
        return index


def _test_different_sequences(num_sequences: int=100):
    for nw in range(2):
        for seed in range(10):
            ds = _IntegerDataset(length=25)
            dl = initialize_train_dataloader(ds, 
                                        seed=seed, 
                                        batch_size=5, 
                                        num_workers=nw, 
                                        warning=False)

            seed_sequences = []

            for _ in range(num_sequences):
                seq = []
                for batch in dl:
                    seq.extend([b.item() for b in batch])
                seed_sequences.append(seq)
            
            # make sure the seed sequences are different
            sim_count = 0
            for s1 in seed_sequences:
                for s2 in seed_sequences:
                    sim_count += int(s1 == s2)

            assert sim_count < 2 * num_sequences, "Too many repeated sequences"


def _test_reproducibility_across_runs(num_sequences: int=10):
    for nw in range(2):
        for seed in range(10):
            sequences = []
            
            for _ in range(num_sequences):
                ds = _IntegerDataset(length=25)
                dl = initialize_train_dataloader(ds, 
                                            seed=seed, 
                                            batch_size=5, 
                                            num_workers=nw, 
                                            warning=False)
                
                seq = []
                for batch in dl:
                    seq.extend([b.item() for b in batch])

                sequences.append(seq)
            
            # make sure the sequences are all the same
            for s in sequences:
                assert s == sequences[0], "The dataloader does not return the same sequence across multiple runs"

if __name__ == '__main__':
    _test_different_sequences()
    _test_reproducibility_across_runs()
