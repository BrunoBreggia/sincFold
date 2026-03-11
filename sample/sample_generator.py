import numpy as np
from sincfold.utils import bp2dot

pairs = {'A': 'U',
         'U': 'A',
         'G': 'C',
         'C': 'G'}

def rna_generator(tree):
    """
    Arma secuencias de ARN ficticias con las estructura solicitada
    en formato arbol (notacion de Newick). 
    Devuelve secuencia de ARN como string mas secuencia punto-parentesis.
    """
    seq_id = tree[0] # root is sequence ID
    pair_array = []
    seq = _rna_concat(tree[1], pair_array)
    pair_array = pair_array[1:]
    return seq_id, seq, pair_array

def _rna_concat(tree, pair_array=None):
    rna = ''
    if len(pair_array)==0:
        pair_array.append(0)
    
    for node in tree:
        if type(node) is int:
            # it is a terminal node
            rna += "".join(np.random.choice(list(pairs.keys()), node)) # insert random nt sequence
            pair_array[0] += node # increase nt counter
        else:
            # it is a subtree
            pos = pair_array[0]
            paired_seq = np.random.choice(list(pairs.keys()), node[0]) # choose random nt sequence
            pair_array[0] += node[0]
            rna += "".join(paired_seq)

            sub_seq = _rna_concat(node[1], pair_array)
            rna += sub_seq

            rna += "".join([pairs[nt] for nt in paired_seq[::-1]])
            pair_array[0] += node[0]

            for i in range(len(paired_seq)):
                pair_array.append([pos+i+1, pos+i+2*(len(paired_seq)-i)+len(sub_seq)])
    return rna

def random_rna_tree():
    """
    Generates random RNA trees
    """
    tree = []

    num = np.random.randint(10,20)
    if num <= 10:
        tree.append(num)
    else:
        num2 = np.random.randint(5,20)
        if num2 <= 10:
            tree.append(num2)

        stem = np.random.randint(3,5)
        subtree = random_rna_tree()
        tree.append((stem, subtree))
    
        num2 = np.random.randint(5,20)
        if num2 < 10:
            tree.append(num2)
    return tree

def create_mock_file(filename):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)

        for i in range(200):
            id, seq, pairs_array = rna_generator([f'prueba{i}', random_rna_tree()])
            if len(seq) < 300 and len(pairs_array) > 0:
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)


if __name__ == '__main__':
    # create_mock_file("sample/train_mock.csv")
    seq_id, seq, pairs_list = rna_generator(["prueba", random_rna_tree()])
    print(seq_id)
    print(seq)
    print(pairs_list)
    dot = bp2dot(pairs_list, len(seq))
    print(dot)
    
