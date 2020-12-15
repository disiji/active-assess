Active Bayesian Assessment for Black-Box Classifiers
===


## Getting started
To setup virtual environment install dependencies in `requirements.txt`:
```
conda create -n active-assessment python=3.7
source activate active-assessment
pip install -r requirements.txt
```


## Data
---
Datasets are uploaded as Appendices. See the main paper and the technical appendices for detailed description of the datasets and the prediction models. Download the datasets and update `DATA_DIR`, `RESULTS_DIR` and `FIGURE_DIR` and `src/data_utils.py` accordingly. 


## Run the active Bayesian assessor
---
Our implementation supports different methods we reported in the paper, including `UPrior`, `IPrior`, `UPrior+TS`, `IPrior+TS` for different assessment tasks:
- `assess_accuracy.py`: train the assessment models for accuracy-related tasks;
- `assess_confusion.py`: train the assessment models for confusion matrix-related tasks;
- `assess_difference.py`: train the assessment models for performance comparison.

For example, to estimate the `groupwise_accuracy` of `cifar-100` with prior strength $$\alpha_k + \beta_k=2$$, navigate to `src` directory and run:
```{bash}
python assess_accuracy.py --dataset_name cifar100 \
                          --group_method predicted_class \
                          --metric groupwise_accuracy \
                          --pseudocount 2 \
                          --topk 1
```

## Eval the assessment methods
We provide the scripts to evaluate the performance of difference methods in terms of different tasks, including:
- `eval_accuracy.py`: this script is for evaluating (1) estimating of groupwise accuracies and (2) identification of extreme classes;
- `eval_confusion.py`: this script is for evaluating estimating of confusion matrices;
- `eval_difference.py`: this script is for evaluating comparison of groupwise accuracies.
For example, to assess the performance of different assessment models for estimating the confusion matrices:
```{bash}
declare -a DatasetNames=("cifar100" "dbpedia" "20newsgroup" "svhn")
for dataset_name in "${DatasetNames[@]}"
do
    python eval_confusion.py --dataset_name $dataset_name &
done
```


## To reproduce the results reported in the paper and supplements:
- `cd src`
- `bash script`

