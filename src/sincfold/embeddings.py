import torch as tr
from itertools import product

# Mapping of nucleotide symbols
# R	Guanine / Adenine (purine)
# Y	Cytosine / Uracil (pyrimidine)
# K	Guanine / Uracil
# M	Adenine / Cytosine
# S	Guanine / Cytosine
# W	Adenine / Uracil
# B	Guanine / Uracil / Cytosine
# D	Guanine / Adenine / Uracil
# H	Adenine / Cytosine / Uracil
# V	Guanine / Cytosine / Adenine
# N	Adenine / Guanine / Cytosine / Uracil
NT_DICT = {
    "R": ["G", "A"],
    "Y": ["C", "U"],
    "K": ["G", "U"],
    "M": ["A", "C"],
    "S": ["G", "C"],
    "W": ["A", "U"],
    "B": ["G", "U", "C"],
    "D": ["G", "A", "U"],
    "H": ["A", "C", "U"],
    "V": ["G", "C", "A"],
    "N": ["G", "A", "C", "U"],
}

VOCABULARY = ["A", "C", "G", "U"]


def generate_kmer_vocabulary(k=3):
    """Generate all possible k-mers from the nucleotide vocabulary.
    
    For k=3, this generates 4^3 = 64 possible 3-mers (AAA, AAC, ACG, ..., UUU)
    
    Args:
        k: Size of the k-mer (default 3)
    
    Returns:
        List of all possible k-mers
    """
    return ["".join(p) for p in product(VOCABULARY, repeat=k)]


KMER_VOCABULARY = generate_kmer_vocabulary(3)  # 64 3-mers
KMER_VOCABULARY.append("PAD")  # Padding token for sequences shorter than k


class OneHotEmbedding:
    def __init__(self):
        self.pad_token = "-"
        self.vocabulary = VOCABULARY
        self.emb_size = len(self.vocabulary)

    def seq2emb(self, seq, pad_token="-"):
        """One-hot representation of seq nt in vocabulary.  Emb is CxL
        Other nt are mapped as shared activations.
        """
        seq = seq.upper().replace("T", "U")  # convert to RNA
        emb_size = len(VOCABULARY)
        emb = tr.zeros((emb_size, len(seq)), dtype=tr.float)

        for k, nt in enumerate(seq):
            if nt == pad_token:
                continue
            if nt in VOCABULARY:
                emb[VOCABULARY.index(nt), k] = 1
            elif nt in NT_DICT:
                v = 1 / len(NT_DICT[nt])
                ind = [VOCABULARY.index(n) for n in NT_DICT[nt]]
                emb[ind, k] = v
            else:
                raise ValueError(f"Unrecognized nucleotide {nt}")

        return emb


class KMerEmbedding:
    """K-mer embedding for RNA sequences.
    
    This class converts an RNA sequence into k-mer tokens and creates a one-hot
    encoding for each k-mer. For k=3, a sequence of length L becomes a sequence
    of length L-k+1 (e.g., "AUGC" -> ["AUG", "UGC"]).
    
    Example:
        >>> embedder = KMerEmbedding(k=3)
        >>> seq = "AUGC"
        >>> emb, token_len = embedder.seq2emb(seq)
        >>> # emb shape: (vocab_size, L-k+1) = (65, 2)
        >>> # token_len = 2 (number of 3-mers)
    """
    
    def __init__(self, k=3):
        """Initialize the k-mer embedder.
        
        Args:
            k: Size of the k-mer (default 3)
        """
        self.k = k
        self.pad_token = "PAD"
        self.vocabulary = KMER_VOCABULARY if k == 3 else generate_kmer_vocabulary(k)
        self.vocabulary.append(self.pad_token)
        self.emb_size = len(self.vocabulary)
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.vocabulary)}
    
    def seq2kmer(self, seq):
        """Convert a sequence to a list of k-mers.
        
        Args:
            seq: RNA sequence string
        
        Returns:
            List of k-mer tokens
        """
        seq = seq.upper().replace("T", "U")  # convert to RNA
        kmers = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k]
            # Handle ambiguous nucleotides by using NNN or skipping
            # Here we just take the exact kmer
            if all(c in VOCABULARY for c in kmer):
                kmers.append(kmer)
            else:
                # Replace ambiguous with NNN (will be handled as unknown)
                kmers.append("N" * self.k)
        return kmers
    
    def seq2emb(self, seq):
        """Convert sequence to k-mer one-hot embedding.
        
        Args:
            seq: RNA sequence string
        
        Returns:
            Tuple of (embedding tensor [vocab_size, L-k+1], token length)
        """
        seq = seq.upper().replace("T", "U")
        kmers = self.seq2kmer(seq)
        L_k = len(kmers)  # contracted length
        
        emb = tr.zeros((self.emb_size, L_k), dtype=tr.float)
        
        for idx, kmer in enumerate(kmers):
            if kmer in self.kmer_to_idx:
                emb[self.kmer_to_idx[kmer], idx] = 1.0
            else:
                # Unknown k-mer - could assign to a special token or skip
                # For now, leave as zeros (effectively ignoring)
                pass
        
        return emb, L_k
