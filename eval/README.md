Codes for evaluation. They are not in src/ because they have nothing to do with taxonomy construction itself.

# Usage
At `HyperMine/` directory, run `python3 eval/evaluation.py <input_file>`.

# Input file 
The input file should contain the hypernym-hyponym pairs and the hypernymy score predicted by your model.
Format:
Each line is `hypernym \t hyponym \t score`. 

# Golden standard
In `HyperMine/config.py`, the following configs should be set

`EvaluationConfig.pos_ancestor_descendant`: should contain the positive ancestor-descendant pairs. Each line is `ancestor \t descendant`.

`EvaluationConfig.pos_parent_child`: should contain the positive parent-child pairs. Each line is `parent \t child`.

`EvaluationConfig.neg_ancestor_descendant`: should contain the negative ancestor-descendant pairs. Each line is `ancestor \t descendant` (but they do not have ancestor-descendant relation, since they are intended to be negative examples).

`EvaluationConfig.neg_parent_child`: should contain the negative parent-child pairs. Each line is `parent \t child` (but they do not have parent-child relation, since they are intended to be negative examples).
