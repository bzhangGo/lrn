## Named Entity Recognition


We employ the birnn plus CRF architecture as [Lample et al. 2016](https://www.aclweb.org/anthology/N16-1030), and
experiment on CoNLL-2003 English NER data.
Main experimental results are summarized below.

<table>
  <tr>
    <th>Model</th>
    <th>#Params</th>
    <th>NER</th>
  </tr>
  <tr>
    <td>Lample et al. 2016</td>
    <td>-</td>
    <td>90.94</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>245K</td>
    <td>89.61</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>192K</td>
    <td>89.35</td>
  </tr>
  <tr>
    <td>ATR</td>
    <td>87K</td>
    <td>88.46</td>
  </tr>
  <tr>
    <td>SRU</td>
    <td>161K</td>
    <td>88.89</td>
  </tr>
  <tr>
    <td>LRN</td>
    <td>129K</td>
    <td>88.56</td>
  </tr>
</table>

F1-score.

## Requirement
see [requirements.txt](code/requirements.txt) for full list.

## How to Run?

- download and preprocess dataset

  - download the conll2003 dataset from [anago](https://github.com/Hironsan/anago/tree/master/data) (in data folder).
  - download the Glove-6B-100d pre-trained word embedding from: http://nlp.stanford.edu/data/glove.6B.zip 

- no hyperparameters are tuned, we keep them all in default.

- training and evaluation

  the running procedure is as follows:
  ```
  export CUDA_ROOT=XXX
  export PATH=$CUDA_ROOT/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

  export CUDA_VISIBLE_DEVICES=0

  export data_dir=path-of/conll2003/en/ner
  export glove_dir=path-of/glove.6B/glove.6B.100d.txt

  RUN_EXP=5
  rnn=lrn

  for i in $(seq 1 $RUN_EXP); do 
      exp_dir=exp$i
      mkdir $exp_dir
      cd $exp_dir

      export cell_type=$rnn
      python3 ner_glove.py --cell lrn >& log.lrn
  
      cd ../
  done

  python scripts/get_test_score.py $rnn exp* >& score.$rnn
  ```
  Results are reported over 5 runs.

## Credits

Source code structure is adapted from [annago](https://github.com/Hironsan/anago/tree/master/).