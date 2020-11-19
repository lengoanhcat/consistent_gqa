# Logically Consistent Loss for Visual Question Answer

This code repository contains codes and scripts for running experiments,
mentioned in the following paper.

[*Logically Consistent Loss for Visual Questions Answering*, Anh-Cat Le-Ngo,
Truyen Tran, Santu Rana, Sunil Gupta, Svetha Venkatesh](docs/paper.pdf)
## Key Logics
1. C-rule:
```python
    if 'C2' in config.tflRules:
        constraints.append(
            tfl.constraint(
                "forall p: forall q: (isDiff(p,q) and isGoodTask(p,q)) -> (isGoodAns(p,q))")
            /tf.cast(tf.square(self.batchSize),tf.float32))
        numCW2 += 1
```

2. E-rule:
```python
    constraints_str = [
        "queryObj(p) -> queryAttrObj(q)",
        "queryAttrObj(p) -> existAttrTrue(q)",
        "existAttrTrue(p) -> existAttrOrTrue(q)",
        "existAttrTrue(p) -> existNotTrue(q)",
        "existAttrTrue(p) -> existAttrNotTrue(q)",
        "existAttrOrTrue(p) -> existNotOrTrue(q)",
        "existNotOrTrue(p) -> existOrTrue(q)",
        "existOrTrue(p) -> existTrue(q)",
        "queryAttrObj(p) -> queryObj(q)",
        "queryNotObj(p) -> existNotTrue(q)",
        "existNotTrue(p) -> existTrue(q)",
        "existAttrNotTrue(p) -> existTrue(q)",
        "existAndTrue(p) -> existTrue(q)",
        "existRelTrue(p) -> existTrue(q)",
        "existRelTrue(p) <-> verifyRelTrue(q)",
        "verifyRelTrue(p) <-> queryRel(q)",
        "verifyRelTrue(p) <-> chooseRel(q)",

        "existOrFalse(p) -> existFalse(q)",

        "existFalse(p) -> existNotFalse(q)",
        "existFalse(p) -> existAttrNotFalse(q)",

        "existFalse(p) -> existRelFalse(q)",
        "existFalse(p) -> existAndFalse(q)",

        "existNotFalse(p) -> existAttrFalse(q)",
        "existAttrNotFalse(p) -> existAttrFalse(q)",

        "existNotOrFalse(p) -> existNotFalse(q)",
        "existNotOrFalse(p) -> existAttrNotFalse(q)",

        "existNotOrFalse(p) -> existAttrOrFalse(q)",
        "existAttrOrFalse(p) -> existAttrFalse(q)",

        "verifyAttrsTrue(p) -> verifyAttrTrue(q)",
        "verifyAttrAndTrue(p)-> verifyAttrTrue(q)",
        "verifyAttrTrue(p) -> queryAttr(q)",
        "queryAttr(p) -> verifyAttrFalse(q)",
        "queryAttr(p) -> chooseAttr(q)",
        "verifyAttrFalse(p) -> verifyAttrAndFalse(q)",
        "verifyAttrAndFalse(p) -> chooseAttr(q)",
        "chooseAttr(p) <-> chooseObj(q)",

        "verifyGlobalTrue(p) -> verifyGlobalFalse(q)",
        "verifyGlobalTrue(p) <-> queryGlobal(q)",
        "verifyGlobalFalse(p) -> chooseGlobal(q)",

        "compare(p) -> common(q)",
        "common(p) -> twoSameTrue(q)",
        "twoSameTrue(p) <-> twoDiffFalse(q)",

        "twoSameFalse(p) <-> twoDiffTrue(q)",

        "allSameTrue(p) <-> allDiffFalse(q)",
        "allSameFalse(p) <-> allDiffTrue(q)"
    ]

    if 'E4' in config.tflRules:
        entailment_str = '({})'.format(constraints_str[0]) + ''.join([
            ' or ({})'.format(task) for task in constraints_str[1:]])
        constraints.append(
            tfl.constraint(
                "forall p: forall q: (isDiff(p,q) and ({})) ->  isGoodAns(p,q)".format(entailment_str))\
            /tf.cast(tf.square(self.batchSize),tf.float32))
        numCW2 += 1
```


## Prerequisites

1. Check out the MACnet architecture for VQA [MAC](https://github.com/stanfordnlp/mac-network/tree/gqa)
2. Follow the setup instructure for downloading and preprocessing **original**
   data

## Hybrid Data Preprocess
1. Setup a mongodb server (>= 4.2.3) by following this
   [instruction](https://docs.mongodb.com/manual/installation/)
2. Use the bash script [import_gqa_into_mongodb.sh](./gqa_preprocess/import_gqa_into_mongodb.sh') to
   import GQA questions and scenegraphs database from the following json files
   into MongoDB collections with their correspondent names
   - train_balanced_questions.json
   - train_all_questions/
   - train_sceneGraphs.json
   - val_balanced_questions.json
   - val_all_questions.json
   - val_sceneGraphs.json
   - testdev_balanced_questions.json
   - testdev_all_questions.json
   - test_balanced_questions.json
   - test_all_questions.json

3. Run the Python Script [generate_multilabel_data_parallel.py](./data/generate_multilabel_data_parallel.py) to
   generate 16 partitions of family-batch data for train dataset
4. Run an experiment with run_experiment.sh

## Analyze prediction after running experiment
1. Use [extract_preds.py](./data/extract_preds.py) to extract predictions from valPredictions.json file into a simpler dictionary
   structure of {qid: answer} in txt files i.e. original.txt, hybrid.txt and
   logic.txt 
2. Use Jupyter notebook [postexp_analysis.ipynb](./preds_postprocess/postexp_analysis.ipynb) to analyze answers in family-batch
