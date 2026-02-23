# imports
import os
import subprocess as sp 
from platform import system
import warnings
import numpy as np
import torch as tr
import pandas as pd

from sincfold.embeddings import NT_DICT
from sincfold import __path__ as sincfold_path
from sincfold.embeddings import NT_DICT, VOCABULARY


def unpool_kmer_matrix(contracted_matrix, L, k=3):
    """Unpool a k-mer level contact matrix back to nucleotide resolution.
    
    This function expands a contact matrix of size (L-k+1) x (L-k+1) to the 
    full nucleotide resolution L x L. Each entry in the contracted matrix 
    represents interactions between two k-mers, which span k nucleotides each.
    
    The expansion uses averaging: each nucleotide position (i, j) in the 
    expanded matrix receives the average of all contracted matrix entries 
    whose k-mer ranges cover (i, j).
    
    Args:
        contracted_matrix: Tensor of shape [batch, L_k, L_k] where L_k = L - k + 1
        L: Original sequence length (nucleotide resolution)
        k: K-mer size (default 3)
    
    Returns:
        Expanded matrix of shape [batch, L, L]
    
    Example:
        For sequence "AUGC" (L=4) with k=3:
        - K-mer positions: 0->"AUG", 1->"UGC" (L_k = 2)
        - contracted[0,1] = interaction between "AUG" and "UGC"
        - This should fill expanded[0:3, 1:4] (positions 0-2 and 1-3)
        
        For overlapping regions, values are averaged.
    """
    batch_size = contracted_matrix.shape[0]
    L_k = contracted_matrix.shape[1]  # contracted length = L - k + 1
    
    # Initialize output matrix with zeros
    expanded = tr.zeros((batch_size, L, L), dtype=contracted_matrix.dtype, 
                        device=contracted_matrix.device)
    
    # Count how many contracted values contribute to each expanded cell
    # This is used for averaging
    count_matrix = tr.zeros((batch_size, L, L), dtype=tr.float32, 
                           device=contracted_matrix.device)
    
    # For each contracted position (i, j), map it to the expanded region
    # Contracted position i corresponds to nucleotide positions [i, i+k)
    for i in range(L_k):
        for j in range(L_k):
            # The k-mer at position i covers nucleotides [i, i+k)
            # The k-mer at position j covers nucleotides [j, j+k)
            # The interaction fills the submatrix [i:i+k, j:j+k]
            i_start, i_end = i, min(i + k, L)
            j_start, j_end = j, min(j + k, L)
            
            expanded[:, i_start:i_end, j_start:j_end] += contracted_matrix[:, i:i+1, j:j+1]
            count_matrix[:, i_start:i_end, j_start:j_end] += 1.0
    
    # Avoid division by zero
    count_matrix = count_matrix.clamp(min=1.0)
    expanded = expanded / count_matrix
    
    return expanded


def normalize_brackets(struct):
    """Unify bracket notation"""
    for b in BRACKET_DICT:
        struct = struct.replace(b, BRACKET_DICT[b])
    return struct


def bracket_match(struct):
    match = True
    for pair in MATCHING_BRACKETS:
        match = match & (struct.count(pair[0]) == struct.count(pair[1]))
    return match


def fold2bp(struc, xop="(", xcl=")"):
    """Get base pairs from one page folding (using only one type of brackets).
    BP are 1-indexed"""
    openxs = []
    bps = []
    if struc.count(xop) != struc.count(xcl):
        return False
    for i, x in enumerate(struc):
        if x == xop:
            openxs.append(i)
        elif x == xcl:
            if len(openxs) > 0:
                bps.append([openxs.pop() + 1, i + 1])
            else:
                return False
    return bps


def dot2bp(struc):
    bp = []
    if not set(struc).issubset(
        set(["."] + [c for par in MATCHING_BRACKETS for c in par])
    ):
        return False

    for brackets in MATCHING_BRACKETS:
        if brackets[0] in struc:
            bpk = fold2bp(struc, brackets[0], brackets[1])
            if bpk:
                bp = bp + bpk
            else:
                return False
    return list(sorted(bp))


def dot2matrix(dot):
    matrix = tr.zeros((len(dot), len(dot)))
    base_pairs = dot2bp(dot)

    for bp in base_pairs:
        # base pairs are 1-based
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1

    return matrix


def bp2matrix(L, base_pairs):
    matrix = tr.zeros((L, L))

    for bp in base_pairs:
        # base pairs are 1-based
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1

    return matrix


def read_ct(ctfile):
    """Read ct file, return sequence and base_pairs"""
    seq, bp = [], []
    
    k = 1
    for p, line in enumerate(open(ctfile)):
        if p == 0:
            try:
                seq_len = int(line.split()[0])
            except ValueError:
                # >seq length: N extra info
                if line.split(":")[0] == ">seq length":
                    seq_len = int(line.split(":")[1].split()[0])
            
            continue 

        if line[0] == "#" or len(line.strip()) == 0:
            # comment
            continue

        line = line.split()
        if len(line) != 6 or not line[0].isnumeric() or not line[4].isnumeric:
            # header
            continue

        n1, n2 = int(line[0]), int(line[4])
        if k != n1: # add missing nucleotides as N
            seq += ["N"] * (n1-k)
        seq.append(line[1])
        k = len(seq) + 1
        if n2 > 0 and (n1 < n2):
            bp.append([n1, n2])

    assert len(seq) == seq_len, f"ct file format error\n{seq_len}\n{seq}\n{len(seq)}"
    return "".join(seq), bp


def write_ct(fname, seqid, seq, base_pairs):
    """Write ct file from sequence and base pairs. Base_pairs should be 1-based and unique per nt"""
    base_pairs_dict = {}
    for bp in base_pairs:
        base_pairs_dict[bp[0]] = bp[1]
        base_pairs_dict[bp[1]] = bp[0]

    with open(fname, "w") as fout:
        fout.write(f"{len(seq)} {seqid}\n")
        for k, n in enumerate(seq):
            fout.write(f"{k+1} {n} {k} {k+2} {base_pairs_dict.get(k+1, 0)} {k+1}\n")


def split_fasta_rec(s, mfe=True):
    """This assume the format of the fasta record is AACCGGUU((....))(-1.2), where the last 
    parenthesis part is optional (mfe)"""
    s = s.strip()
    mfe_start = s.rfind("(")

    if mfe:
        mfe = float(s[mfe_start + 1 : -1])
    s = s[:mfe_start].strip()

    seq = s[: len(s) // 2]
    struct = s[len(s) // 2 :]
    
    assert len(seq) == len(struct), "Sequence and structure have different lengths"

    return seq, struct, mfe


def mat2bp(x):
    """Get base-pairs from conection matrix [N, N]. It uses upper
    triangular matrix only, without the diagonal. Positions are 1-based. """
    ind = tr.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = tr.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = tr.where(pairs_ind[:, 1]==i)[0]
        if len(ind)>0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = tr.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()
                
            multiplets += [pairs[k] for k in range(len(pairs)) if k!=best_pair]   
            
    pairs_ind = [[bp[0]+1, bp[1]+1] for bp in pairs_ind.tolist() if bp not in multiplets]
 
    return pairs_ind


def postprocessing(preds, masks):
    """Postprocessing function using viable pairing mask.
    Inputs are batches of size [B, N, N]"""
    if masks is not None:
        preds = preds.multiply(masks)

    y_pred_mask_triu = tr.triu(preds)
    y_pred_mask_max = tr.zeros_like(preds)
    for k in range(preds.shape[0]):
        y_pred_mask_max_aux = tr.zeros_like(y_pred_mask_triu[k, :, :])

        val, ind = y_pred_mask_triu[k, :, :].max(dim=0)
        y_pred_mask_max[k, ind[val > 0], val > 0] = val[val > 0]

        val, ind = y_pred_mask_max[k, :, :].max(dim=1)
        y_pred_mask_max_aux[val > 0, ind[val > 0]] = val[val > 0]

        ind = tr.where(y_pred_mask_max[k, :, :] != y_pred_mask_max_aux)
        y_pred_mask_max[k, ind[0], ind[1]] = 0

        y_pred_mask_max[k] = tr.triu(y_pred_mask_max[k]) + tr.triu(
            y_pred_mask_max[k]
        ).transpose(0, 1)
    return y_pred_mask_max

def find_pseudoknots(base_pairs):
    pseudoknots = []
    for i, j in base_pairs:
        for k, l in base_pairs:
            if i < k < j < l:  # pseudoknot definition
                if [k, l] not in pseudoknots:
                    pseudoknots.append([k, l])
    return pseudoknots

def dot2png(png_file, sequence, dotbracket, resolution=10):

    try:
        sp.run("java -version", shell=True, check=True, capture_output=True)
        sp.run(f'java -cp {VARNA_PATH} fr.orsay.lri.varna.applications.VARNAcmd -sequenceDBN {sequence} -structureDBN "{dotbracket}" -o  {png_file} -resolution {resolution}', shell=True)
    except:
        warnings.warn("Java Runtime Environment failed trying to run VARNA. Check if it is installed.")
    
    
def ct2svg(ct_file, svg_file):
    
    sp.run(f'{DRAW_CALL} {ct_file} {svg_file}', shell=True, capture_output=True)


def ct2dot(ct_file):
    if not os.path.isfile(ct_file) or os.path.splitext(ct_file)[1] != ".ct":
        raise ValueError("ct2dot requires a .ct file")
    dotbracket = ""
    if CT2DOT_CALL:
        sp.run(f"{CT2DOT_CALL} {ct_file} 1 tmp.dot", shell=True, capture_output=True)
        try: 
            dotbracket = open("tmp.dot").readlines()[2].strip()
            os.remove("tmp.dot")
        except FileNotFoundError: 
            print("Error in ct2dot: check .ct file")
    else:
        print("Dotbracket conversion only available on linux")
    return dotbracket


def valid_sequence(seq):
    """Check if sequence is valid"""
    return set(seq.upper()) <= (set(NT_DICT.keys()).union(set(VOCABULARY)))

def validate_file(pred_file):
    """Validate input file fasta/csv format and return csv file"""
    if os.path.splitext(pred_file)[1] == ".fasta":
        table = []
        with open(pred_file) as f:
            row = [] # id, seq, (optionally) struct
            for line in f:
                if line.startswith(">"):
                    if row:
                        table.append(row)
                        row = []
                    row.append(line[1:].strip())
                else:
                    if len(row) == 1: # then is seq
                        row.append(line.strip())
                        if not valid_sequence(row[-1]):
                            raise ValueError(f"Sequence {row.upper()} contains invalid characters")
                    else: # struct
                        row.append(line.strip()[:len(row[1])]) # some fasta formats have extra information in the structure line
        if row:
            table.append(row)
        
        pred_file = pred_file.replace(".fasta", ".csv")
        
        if len(table[-1]) == 2:
            columns = ["id", "sequence"]
        else:
            columns = ["id", "sequence", "dotbracket"]

        pd.DataFrame(table, columns=columns).to_csv(pred_file, index=False)

    elif os.path.splitext(pred_file)[1] != ".csv":
        raise ValueError("Predicting from a file with format different from .csv or .fasta is not supported")
    
    return pred_file 

def validate_canonical(sequence, base_pairs):
    if not valid_sequence(sequence):
        return False, "Invalid sequence"

    for i, j in base_pairs:
        nt1, nt2 = sequence[i-1], sequence[j-1]
        if pair_strength((nt1, nt2))==0:
            return False, f"Invalid base pair: {nt1} {nt2}"

        for k, l in base_pairs:
            if (k, l) != (i, j):
                if i in (k, l):
                    return False, f"Nucleotide {i} is in pair {i, j} and {k, l}"
                if j in (k, l):
                    return False, f"Nucleotide {j} is in pair {i, j} and {k, l}"

    return True, ""