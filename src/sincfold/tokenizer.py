import torch as tr
import math
from itertools import product

# This order matters!
VOCABULARY = ["A", "C", "G", "U"]

KMERVOCAB = [""]
KMERVOCAB += ["".join(p) for p in product(VOCABULARY, repeat=3)]  # 64 tokens
# Total of 64 elements in 3-mer vocabulary


def k3_tokenizer(seq:str) -> tr.Tensor:
    """ 
    Receives an arbitrarily long (L) RNA sequence and
    returns a sequence of IDs (int) of length L_3 that corresponds
    to the 3-mers of the original sequence.

    Tokenized length is floor(L/3).
    """
    k=3
    num_tokens = len(seq)//3
    tokens = tr.zeros(num_tokens, dtype=tr.int16)

    for i in range(num_tokens):
        tok = seq[k*i : k*(i+1)]
        tok_id = KMERVOCAB.index(tok)
        tokens[i] = tok_id
    
    return tokens


if __name__ == "__main__":
    seq = "AUCCGUUUAGUCUUAGAAUCGAUCGAUC"
    print(seq)
    print(f"Seq len: {len(seq)}")
    print()
    tokenized = k3_tokenizer(seq)
    print(tokenized)
    print(f"Tokenized len: {len(tokenized)}")
    print()

    input_tensor = tr.arange(18, dtype=tr.float).reshape(2, 1, 3, 3)
    target_size = (input_tensor.shape[-1]*3,) *2
    resized_tensor = tr.nn.functional.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)
    print(target_size)
    print()
    print(resized_tensor)

