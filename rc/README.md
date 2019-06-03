## Reading Comprehension


We use [SQUAD v1](https://rajpurkar.github.io/SQuAD-explorer/) for experiments and adopt the 
[RNET model](https://www.aclweb.org/anthology/papers/P/P17/P17-1018/). 
Main experimental results are summarized below.

<table>
  <tr>
    <th>Model</th>
    <th>#Params</th>
    <th>Base</th>
    <th>+Elmo</th>
  </tr>
  <tr>
    <td>rnet</td>
    <td>-</td>
    <td>71.1/79.5</td>
    <td>-/-</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>2.67M</td>
    <td>70.46/78.98</td>
    <td>75.17/82.79</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>2.31M</td>
    <td>70.41/79.15</td>
    <td>75.81/83.12</td>
  </tr>
  <tr>
    <td>ATR</td>
    <td>1.59M</td>
    <td>69.73/78.70</td>
    <td>75.06/82.76</td>
  </tr>
  <tr>
    <td>SRU</td>
    <td>2.44M</td>
    <td>69.27/78.41</td>
    <td>74.56/82.50</td>
  </tr>
  <tr>
    <td>LRN</td>
    <td>2.14M</td>
    <td>70.11/78.83</td>
    <td>76.14/83.83</td>
  </tr>
</table>

Exact match/F1-score.

## Requirement
tensorflow >= 1.8.1

## How to Run?

- download and preprocess dataset

  - see [R-Net](https://github.com/HKUST-KnowComp/R-Net) about the preprocessing of datasets
  - Basically, you need the following datasets: squad v1.1, GloVe, Elmo and convert raw datasets into the required data format.

- no hyperparameters are tuned, we keep them all in default.

- training and evaluation

  Please see the `train_lrn.sh` and `test_lrn.sh` scripts in `rnet` (Base) and `elmo_rnet` (Base+Elmo).
  
  For reporting final EM/F1 score, we used the `evaluate-v1.1.py` script.

## Credits

Source code structure is adapted from [R-Net](https://github.com/HKUST-KnowComp/R-Net).