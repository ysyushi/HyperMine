## Model Train

```
$ python main_train.py -modelName lt_bilinearDIAG -lr 0.01 -device-id 0 -epochs 100 -remark test
$ (pairwise_model) python main_train.py -modelName lt_bilinearDIAG+PF -lr 0.01 -margin 1.0 -batch-size 50 -h1 64 -device-id 0 -epochs 50 -use-pair-feature 1
$ (pairwise_only_model) python main_train.py -modelName lt_PF -lr 0.01 -margin 1.0 -batch-size 50 -h1 64 -pt-name ld -device-id 0 -epochs 200 -use-pair-feature 1
$ (full_model) python main_train.py -modelName ltdl_bilinearDIAG+PTF -node-dropout 0.5 -h1 64 -h2 32 -pt-name ld -edge-dropout 0.5 -device-id 0 -epochs 200 -use-pair-feature 1
```

## Model Tuning

First update the search space in main_tune.py and then run the following command:

```
$ python main_tune.py -device-id 0 -epochs 50 -use-pair-feature 1 -remark dblp-0126
$ python main_tune.py -device-id 6 -use-pair-feature 1 -supervision-file /shared/data/jiaming/linkedin-maple/maple/data/supervision_pairs3.txt -remark dblp-0201
```

## Model Prediction


```
$ python main_predict.py -modelName lt_bilinearDIAG -device-id 0 -snapshot /shared/data/jiaming/linkedin-maple/maple/src/ranking_model/snapshots/Dec30_14-28-32_test/best_steps_10.pt
$ (pairwise_model) python main_predict.py -modelName lt_bilinearDIAG+PF -device-id 0 -use-pair-feature 1 -snapshot /shared/data/jiaming/linkedin-maple/maple/src/ranking_model/snapshots/Jan17_04-07-55_/best_steps_10.pt
```

## Cat positive/negative pairs to one file

Use code in /shared/data/jiaming/linkedin-maple/maple/src/ranking_model/utils-data-conversion.ipynb

## Model Step-by-Step Tuning

Use code in /shared/data/jiaming/linkedin-maple/maple/src/ranking_model/model_step_by_step_tune.ipynb