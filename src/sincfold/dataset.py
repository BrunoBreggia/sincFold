import pandas as pd
from torch.utils.data import Dataset
import torch as tr
import os
import json
import pickle
from sincfold.embeddings import OneHotEmbedding, KMerEmbedding
from sincfold.utils import valid_mask, prob_mat, bp2matrix, dot2bp

class SeqDataset(Dataset):
    def __init__(
        self, dataset_path, min_len=0, max_len=512, verbose=False, cache_path=None, for_prediction=False, 
        interaction_prior="probmat", use_cannonical_mask=False, training=False,
        kmer_embedding=False, kmer_size=3, **kargs):
        """
        Args:
            dataset_path: Path to the CSV dataset
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            verbose: Print progress
            cache_path: Path to cache directory
            for_prediction: If True, don't expect base pairs
            interaction_prior: "none" or "probmat" - whether to include prior probabilities
            use_cannonical_mask: Whether to use canonical base pair mask
            training: Whether in training mode
            kmer_embedding: If True, use k-mer tokenization instead of nucleotide-level
            kmer_size: Size of k-mer (default 3)
        """
        self.max_len = max_len
        self.verbose = verbose
        if cache_path is not None and not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path

        # Loading dataset
        data = pd.read_csv(dataset_path)
        self.training = training

        if for_prediction:
            assert (
                "sequence" in data.columns
                and "id" in data.columns
            ), "Dataset should contain 'id' and 'sequence' columns"

        else:
            assert (
                ("base_pairs" in data.columns or "dotbracket" in data.columns)
                and "sequence" in data.columns
                and "id" in data.columns
            ), "Dataset should contain 'id', 'sequence' and 'base_pairs' or 'dotbracket' columns"

            if "base_pairs" not in data.columns and "dotbracket" in data.columns:
                data["base_pairs"] = data.dotbracket.apply(lambda x: str(dot2bp(x)))      

        data["len"] = data.sequence.str.len()

        if max_len is None:
            max_len = max(data.len)
        self.max_len = max_len

        datalen = len(data)

        data = data[(data.len >= min_len) & (data.len <= max_len)]

        if len(data) < datalen:
            print(
                f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
            )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        
        # Choose embedding type based on kmer_embedding flag
        self.kmer_embedding = kmer_embedding
        self.kmer_size = kmer_size
        
        if kmer_embedding:
            # Use k-mer embedding (k=3 by default)
            self.embedding = KMerEmbedding(k=kmer_size)
        else:
            # Use original nucleotide-level one-hot embedding
            self.embedding = OneHotEmbedding()
        
        self.embedding_size = self.embedding.emb_size
        self.interaction_prior = interaction_prior
        self.use_cannonical_mask = use_cannonical_mask

        self.base_pairs = None
        if "base_pairs" in data.columns:
            self.base_pairs = [
                json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
            ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
        cache = f"{self.cache}/{seqid}.pk"
        if (self.cache is not None) and os.path.isfile(cache):
            item = pickle.load(open(cache, "rb"))
        else:
            sequence = self.sequences[idx]
            L = len(sequence)  # Original nucleotide length
            L_k = L - self.kmer_size + 1  # K-mer contracted length
            
            Mc = None
            if self.base_pairs is not None:
                Mc = bp2matrix(L, self.base_pairs[idx])

            # Get embedding - k-mer embedding returns (emb, token_len) tuple
            if self.kmer_embedding:
                seq_emb, token_len = self.embedding.seq2emb(sequence)
            else:
                seq_emb = self.embedding.seq2emb(sequence)
                token_len = L  # For nucleotide embedding, token length = sequence length

            mask = None
            if self.use_cannonical_mask:
                mask = valid_mask(sequence)
            interaction_prior = None
            if self.interaction_prior == "probmat":
                interaction_prior = prob_mat(sequence)
            
            item = {
                "embedding": seq_emb, 
                "contact": Mc, 
                "length": L, 
                "token_length": token_len,  # Contracted k-mer length
                "kmer_size": self.kmer_size,
                "canonical_mask": mask,
                "id": seqid, 
                "sequence": sequence, 
                "interaction_prior": interaction_prior
            } 

            if self.cache is not None:
                pickle.dump(item, open(cache, "wb"))
                
        return item

def pad_batch(batch):
    """batch is a dictionary with different variables lists
    
    This function pads sequences to the maximum length in the batch.
    For k-mer embedding, the embedding is padded to max token length (L-k+1).
    """
    
    L = [b["length"] for b in batch]
    L_k = [b["token_length"] for b in batch]  # K-mer contracted lengths
    kmer_size = batch[0]["kmer_size"]
    
    # Pad embedding based on k-mer token length
    # For k-mer: max(L_k) = max(L) - k + 1
    # For nucleotide: L_k == L
    max_L_k = max(L_k)
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], max_L_k))
    
    if batch[0]["contact"] is None:
        contact_pad = None
    else:
        contact_pad = -tr.ones((len(batch), max(L), max(L)), dtype=tr.long)
    
    if batch[0]["canonical_mask"] is None:
        canonical_mask_pad = None
    else:
        canonical_mask_pad = tr.zeros((len(batch), max(L), max(L)))
    
    interaction_prior_pad = None
    if batch[0]["interaction_prior"] is not None:
        interaction_prior_pad = tr.zeros((len(batch), max(L), max(L)))

    for k in range(len(batch)):
        # Embedding is padded to token length (L_k)
        embedding_pad[k, :, : L_k[k]] = batch[k]["embedding"]
        if contact_pad is not None:
            contact_pad[k, : L[k], : L[k]] = batch[k]["contact"]
        if canonical_mask_pad is not None:
            canonical_mask_pad[k, : L[k], : L[k]] = batch[k]["canonical_mask"]
        
        if interaction_prior_pad is not None:
            interaction_prior_pad[k, : L[k], : L[k]] = batch[k]["interaction_prior"]

    out_batch = {
        "contact": contact_pad, 
        "embedding": embedding_pad, 
        "length": L, 
        "token_length": L_k,
        "kmer_size": kmer_size,
        "canonical_mask": canonical_mask_pad,
        "interaction_prior": interaction_prior_pad,
        "sequence": [b["sequence"] for b in batch],
        "id": [b["id"] for b in batch]
    }
    
    return out_batch
