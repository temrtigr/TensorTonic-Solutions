import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    
    N = len(seqs)
    L = max_len if max_len is not None else max(len(seq) for seq in seqs) if seqs else 0
    
    result = np.full((N, L), pad_value)
    
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)      
        result[i, :length] = seq[:length]
    
    return result