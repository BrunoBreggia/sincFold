import torch as tr
import math
from itertools import product

# This order matters!
VOCABULARY = ["A", "C", "G", "U"]

KMERVOCAB = [""]  # 1 token
KMERVOCAB += VOCABULARY[:]  # 4 tokens
KMERVOCAB += ["".join(p) for p in product(VOCABULARY, repeat=2)]  # 16 tokens
KMERVOCAB += ["".join(p) for p in product(VOCABULARY, repeat=3)]  # 64 tokens
# Total of 85 elements in 3-mer vocabulary


def k3_tokenizer(seq:str) -> tr.Tensor:
    """ 
    Receives an arbitrarily long (L) RNA sequence and
    returns a sequence of IDs (int) of length L_3 that corresponds
    to the 3-mers of the original sequence.

    Tokenized length is ceil(L/3).
    """
    k=3
    num_tokens = int(math.ceil(len(seq)/3))
    tokens = tr.zeros(num_tokens, dtype=tr.int16)

    for i in range(num_tokens):
        tok = seq[k*i : k*(i+1)]
        tok_id = KMERVOCAB.index(tok)
        tokens[i] = tok_id
    
    return tokens


def unpool_kmer_matrix(contracted_matrix, L, k=3):
    """Unpool a k-mer level contact matrix back to nucleotide resolution.
    
    This function expands a contact matrix of size ceil(L/3) x ceil(L/3) to the 
    full nucleotide resolution L x L. Each entry in the contracted matrix 
    represents interactions between two k-mers, which span k nucleotides each.
    
    The expansion is done by repetition.
    
    Args:
        contracted_matrix: Tensor of shape [batch, L_k, L_k]
        L: Original sequence length (nucleotide resolution)
        k: K-mer size (default 3)
    
    Returns:
        Expanded matrix of shape [batch, L, L]
    """
    batch_size = contracted_matrix.shape[0]
    L_k = contracted_matrix.shape[1]  # contracted length = L - k + 1
    
    # Initialize output matrix with zeros
    expanded = tr.zeros((batch_size, L, L), dtype=contracted_matrix.dtype, 
                        device=contracted_matrix.device)
    
    # For each contracted position (i, j), map it to the expanded region
    # Contracted position i corresponds to nucleotide positions [i, i+k)
    for i in range(L_k):
        for j in range(L_k):
            expanded[:, k*i:k*(i+1), k*j:k*(j+1)] = contracted_matrix[:, i, j]
    
    return expanded


if __name__ == "__main__":
    seq = "AUCCGUUUAGUCUUAGAAUCGAUCGAUC"
    print(seq)
    print(f"Seq len: {len(seq)}")
    print()
    tokenized = k3_tokenizer(seq)
    print(tokenized)
    print(f"Tokenized len: {len(tokenized)}")

    print()
    mat = tr.randn((1,3,3))
    print(mat)
    expanded = unpool_kmer_matrix(mat,8)
    print(expanded)

