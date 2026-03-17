import numpy as np
from sincfold.utils import bp2dot, dot2bp

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
        if num2 <= 10:
            tree.append(num2)
    return tree

def random_rna_tree_tandem(n=3):
    """
    Generates random RNA trees by tandems of 3 nucleotides
    """
    tree = []

    num = np.random.randint(3,8)
    if num <= 3:
        tree.append(num*n)
    else:
        num2 = np.random.randint(2,6)
        if num2 <= 3:
            tree.append(num2*n)

        stem = np.random.randint(1,2)
        subtree = random_rna_tree_tandem(n)
        tree.append((stem*n, subtree))
    
        num2 = np.random.randint(2,6)
        if num2 <= 3:
            tree.append(num2*n)
    return tree

def create_mock_training_file(filename):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)

        i = 0
        for _ in range(1000):
            id, seq, pairs_array = rna_generator([f'prueba{i}', random_rna_tree_tandem(n=3)])
            if len(seq) < 300 and len(pairs_array) > 0:
                i += 1
                print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

def create_mock_evaluation_file(filename):
    with open(filename, 'w') as samples_file:
        # FASTA file
        for i in range(100):
            id, seq, pairs_array = rna_generator([f'prueba{i}', random_rna_tree_tandem(n=3)])
            if len(seq) < 300 and len(pairs_array) > 0:
                print(">", id, file=samples_file, flush=True)
                print(seq, file=samples_file, flush=True)
                print(bp2dot(pairs_array, len(seq)), file=samples_file, flush=True)

def create_rna_from_structure(structure):
    """
    Create fixed-structure RNA with random nucleotide content
    """
    rna = []
    stack = []
    for i in structure:
        if i == ".":
            rna.append(np.random.choice(list(pairs.keys())))
        elif i == "(":
            nt = np.random.choice(list(pairs.keys()))
            rna.append(nt)
            stack.append(pairs[nt])
        else: # i == ")"
            try:
                rna.append(stack.pop())
            except(IndexError) as e:
                raise e("No coinciden cantidad de parentesis que abren con los que cierran")
    if len(stack) != 0:
        raise IndexError("No coinciden cantidad de parentesis que abren con los que cierran")
    return "".join(rna)


def create_simple_training_file(filename, dot):
    with open(filename, 'w') as samples_file:
        print("id,sequence,base_pairs", file=samples_file, flush=True)

        for i in range(1000):
            id = f"prueba{i}"
            seq = create_rna_from_structure(dot)
            pairs_array = dot2bp(dot)
            print(id, seq, f'"{pairs_array}"', sep=',', file=samples_file, flush=True)

def create_simple_testing_file(filename, dot):
    with open(filename, 'w') as samples_file:
        # FASTA file
        for i in range(100):
            id = f"prueba{i}"
            seq = create_rna_from_structure(dot)
            pairs_array = dot2bp(dot)
            if len(seq) < 300 and len(pairs_array) > 0:
                print(">", id, file=samples_file, flush=True)
                print(seq, file=samples_file, flush=True)
                print(bp2dot(pairs_array, len(seq)), file=samples_file, flush=True)


if __name__ == '__main__':
    # dot1 = "......((((((((((((((((((.................................))))))))))))))))))......(((((((((............)))))))))..."
    dot1 = "((((((.........))))))"
    print(len(dot1))
    create_simple_training_file("sample/train_mock_k3.csv", dot1)
    create_simple_testing_file("sample/test_mock_k3.fasta", dot1)

    # random_rna = random_rna_tree_tandem(3)
    # print(random_rna)
    # seq_id, seq, pairs_list = rna_generator(["prueba", random_rna])
    # print(seq_id)
    # print(seq)
    # print(pairs_list)
    # dot = bp2dot(pairs_list, len(seq))
    # print(dot)

    # dot = "......(((((((((............)))))))))..."
    # rna = create_rna_from_structure(dot)
    # print(rna)
    # pairs_array = dot2bp(dot)
    # print(pairs_array)
