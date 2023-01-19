# A Comparative Survey of Instance Selection Methods applied to NonNeural and Transformer-Based Text Classification

This repository contains a Python 3 implementation of all Instance Selection approaches studied on the proposed ACM Computing Surveys - CSUR journal.

## Installing

Clone this repository in your machine. Execute the installation under the settings directory.

```
git clone https://github.com/waashk/instanceselection.git
cd instanceselection/settings/
bash setup-isenv.sh
```

## Installing Activating environment

```
source isenv/bin/activate
```

### Requirements

This project is based on ```python==3.6```.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Example 

```
python run_generateSplit.py -d <dataset> -m <ismethod> --datain <data_directory> --out <output_directory>;
```

## Supported Instance Selection Methods

The supported Instance Selection Methods are:

- CNN: Condensed Nearest Neighbor (The condensed nearest neighbor rule.)
- ENN: Edited Nearest Neighbor (Asymptotic properties of nearest neighbor rules using edited data.)
- IB3: Instance-Based 3 (Instance-based learning algorithms.)
- DROP3: Decremental Reduction Optimization Procedure #3 (Reduction techniques for instance-based learning algorithms.)
- ICF: Iterative Case Filtering algorithm (Advances in instance selection for instance-based learning algorithms.)
- LSSm: Local Set-based Smoother (Three new instance selection methods based on local sets: A comparative study with several approaches from a bi-objective perspective. P)
- LSBo: Local Set Border Selector (Three new instance selection methods based on local sets: A comparative study with several approaches from a bi-objective perspective. P)
- LDIS: Local Density-based IS (A Density-Based Approach for Instance Selection.)
- CDIS: Central Density-based IS (A Novel Density-Based Approach for Instance Selection.)
- XLDIS: eXtended Local Density-based IS (An Efficient Approach for Instance Selection.)
- PSDSP: Prototype Selection based on Dense Spatial Partitions (Efficient Instance Selection Based on Spatial Abstraction.)
- EGDIS: Enhanced Global Density-based IS (A new approach for instance selection: Algorithms, evaluation, and comparisons. )
- CIS: Curious IS (Curious instance selection.)


## Datasets

To guarantee the reproducibility of the obtained results, all datasets and their respective CV train-test partitions are available on:

All datasets are available at: https://150.164.2.44/datasets/

Each dataset contains the following files:
- texts.txt: Raw document set (text). One per line.
- score.txt: Document class whose index is associated with texts.txt
- splits/split_\<k\>.pkl:  pandas DataFrame with k-cross validation partition.
- tfidf/: the TFIDF representation for each fold in the CSR matrix format. (.gz)

## IS Input Representation 

The TF-IDF representation is used as input to all IS methods. Before creating the TFIDF matrix, we adopted the following steps as a pre-processing step: i. we removed stopwords using the standard list from the scikit-learn library [ 58 ] (version 0.23.2); and ii. we only kept features that appear in at least two documents.

## Output

The outputs of the run_generateSplit.py script are:

- saida_\<method\>.json: A JSON file containing general information about the execution, including i. reduction, ii. time, iii. hyperparameters, among others.
- split_\<nfolds\>_\<method\>.pkl: The new split train-test.
- split_\<nfolds\>_\<method\>_idxinfold.pkl: The training indexes considering the old partition. E.g. Through these indexes, you can directly access the respective documents in a specific fold of the TFIDF representation.


## Citation

```
@article{cunha23,
title = {A Comparative Survey of Instance Selection Methods applied to NonNeural and Transformer-Based Text Classification},
journal = {ACM Computing Surveys - CSUR},
volume = {xx},
number = {xx},
pages = {xxxx},
year = {2023},
issn = {0360-0300},
doi = {https://doi.org/xxxxxx},
url = {https://xxxxx},
author = {Washington Cunha and Felipe Viegas and Celso França and Thierson Rosa and Leonardo Rocha and Marcos André Gonçalves}
}
```

**Note**: This repository contains Unofficial implementations for the methods. 
All IS methods were implemented considering the specification in their respective published papers.
