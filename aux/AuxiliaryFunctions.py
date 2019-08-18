import numpy as np

# Input file with format of word2vec output.
# Parse into a dict of {node: row_index} and a numpy array of embedding vectors
def parse_embedding(input_file):
    with open(input_file, "r") as f_in:
        num_nodes, dim = map(int, f_in.readline().strip().split())
        node_index_dict = {}  # {node: row_index}, where type(node) == str, type(row_index) == int
        embedding_matrix = np.zeros((num_nodes, dim), dtype=np.float_)
        idx = 0
        for line in f_in:
            line_split = line.strip().split()
            cur_name = line_split[0]
            node_index_dict[cur_name] = idx
            embedding_matrix[idx] = np.asarray(line_split[1:], dtype=np.float_)
            idx += 1

    assert idx == num_nodes, "Node number does not match."
    return node_index_dict, embedding_matrix
