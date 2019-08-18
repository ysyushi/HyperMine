# maple
Working repo for the linkedin maple project.




## Weak Supervision I/O Format

**Note: We need to assume phrases in the dictionary are underscore-concatenated (words joined by underscores) to comply with the gensim format to be used, which is not the case for existing data. The following documentation assumes this assumption is true. If Jiaming is building the NN model on top of existing data, please consider this.**

In this section, we refer to a concept/entity as a node when the context is clear considering the use of network data in this project.

The weak supervision model is provided with nodewise features for all nodes, pairwise features for selected node pairs, and a set of *positive* node pairs as weak supervision.

Note that we may provide multiple files for pairwise features and multiple files for nodewise features down the road in order to enable potentially tweaked neural network structures that distinguishes features from different sources (e.g., embedding v.s. DIH-based)/

### Nodewise Feature Format

Each nodewise feature file follows the embedding file format in gensim with the first line being
```
num_records num_features
```
and the each of the rest of line being
```
node_name feature_value_1 feature_value_2 ...
```
where **spaces** are used as separators on each line.

### Pairwise Feature Format

Since pairwise feature is sparse, we use one file for one pairwise feature. In one file, each line is a record for a hypernym-hyponym pair with the format
```
hypernym_node_name	hyponym_node_name	feature_value
```
where **tabs** are used as separators on each line. We use tabs for better readability. 

An example file can be found at `/data/yushi2/linkedin_maple/maple_github/data/one_example_pair_feature.tsv` on dmserv4 (existing as of Dec. 19, 2018).

### Supervision
The supervision file provides a list of **positive** hypernym-hyponym pairs. Each line is a record with format
```
hypernym_node_name	hyponym_node_name
```
where **tabs** are used as separators.

An example file can be found at `/data/yushi2/linkedin_maple/maple_github/data/example_supervision_from_dblp.tsv` on dmserv4 (existing as of Dec. 19, 2018).

### Output Format

We expect the output format to be similar to pairwise feature with each line being
```
hypernym_node_name	hyponym_node_name	score
```
where higher score indicates higher likelihodd. If needed, we may additionally provide a list of pairs to be evaluated for the NN model to make prediction on with a format identical to the supervision file.


