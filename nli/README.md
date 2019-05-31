## natural language inference in LRN model

The dataset is Stanford Natural Language Inference (SNLI), which we regard as a three-way classification tasks. 
We use an encoder-attention-decoder architecture, and stack two additional birnn upon the final sequence representation.
Both GloVe word embedding and character embedding is used for word-level representation.
Main experimental results are summarized below.

 <table>
  <tr>
    <th colspan="2" rowspan="2">Model</th>
    <th rowspan="2">#Params</th>
    <th colspan="2">Base</th>
    <th colspan="2">+LN</th>
    <th colspan="2">+BERT</th>
    <th colspan="2">+LN+BERT</th>
  </tr>
  <tr>
    <td>ACC</td>
    <td>Time</td>
    <td>ACC</td>
    <td>Time</td>
    <td>ACC</td>
    <td>Time</td>
    <td>ACC</td>
    <td>Time</td>
  </tr>
  <tr>
    <td colspan="2">Rocktaschel et al. (2016)</td>
    <td>250K</td>
    <td>83.50</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">This <br>Work</td>
    <td>LSTM</td>
    <td>8.36M</td>
    <td>84.27</td>
    <td>0.262</td>
    <td>86.03</td>
    <td>0.432</td>
    <td>89.95</td>
    <td>0.544</td>
    <td>90.49</td>
    <td>0.696</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>6.41M</td>
    <td>85.71</td>
    <td>0.245</td>
    <td>86.05</td>
    <td>0.419</td>
    <td>90.29</td>
    <td>0.529</td>
    <td>90.10</td>
    <td>0.695</td>
  </tr>
  <tr>
    <td>ATR</td>
    <td>2.87M</td>
    <td>84.88</td>
    <td>0.210</td>
    <td>85.81</td>
    <td>0.307</td>
    <td>90.00</td>
    <td>0.494</td>
    <td>90.28</td>
    <td>0.580</td>
  </tr>
  <tr>
    <td>SRU</td>
    <td>5.48M</td>
    <td>84.28</td>
    <td>0.258</td>
    <td>85.32</td>
    <td>0.283</td>
    <td>89.98</td>
    <td>0.543</td>
    <td>90.09</td>
    <td>0.555</td>
  </tr>
  <tr>
    <td>LRN</td>
    <td>4.25M</td>
    <td>84.88</td>
    <td>0.209</td>
    <td>85.06</td>
    <td>0.223</td>
    <td>89.98</td>
    <td>0.488</td>
    <td>89.93</td>
    <td>0.506</td>
  </tr>
</table>

## How to Run?

- download and preprocess dataset

  - The dataset link: https://nlp.stanford.edu/projects/snli/
  - Prepare separate data files:
    
    We provide a simple processing script `convert_to_plain.py` in scripts folder. By calling:
    ```
    python convert_to_plain.py snli_1.0/[ds].txt
    ```
    you can get the `*.p, *.q, *.l` files as in the `config.py`. *[ds]* indicates `snli_1.0_train.txt`, 
    `snli_1.0_dev.txt` and `snli_1.0_test.txt`. We only preserve `'entailment', 'neutral', 'contradiction'` instances, 
    and others are dropped.
    
  - Prepare embedding and vocabulary
  
    Download the [pre-trained GloVe embedding](http://nlp.stanford.edu/data/glove.840B.300d.zip). And prepare 
    the char as well as word vocabulary using `vocab.py` as follows:
    ```
    # word embedding & vocabulary
    python vocab.py --embeddings [path-to-glove-embedding] train.p,train.q,dev.p,dev.q,test.p,test.q word_vocab
    # char embedding
    python vocab.py --char train.p,train.q,dev.p,dev.q,test.p,test.q char_vocab
    ```
    
   - Download BERT pre-trained embedding (if you plan to work with BERT)

- training and evaluation

  - Train the model as follows:
  ```
  # configure your cuda libaray if necessary
  export CUDA_ROOT=XXX
  export PATH=$CUDA_ROOT/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

  # LRN
  python code/run.py --mode train --config config.py --parameters=gpus=[0],cell="lrn",layer_norm=False,output_dir="train_no_ln" >& log.noln
  # LRN + LN
  python code/run.py --mode train --config config.py --parameters=gpus=[0],cell="lrn",layer_norm=True,output_dir="train_ln" >& log.ln
  # LRN + BERT
  python code/run.py --mode train --config config_bert.py --parameters=gpus=[0],cell="lrn",layer_norm=False,output_dir="train_no_ln_bert" >& log.noln.bert
  # LRN + LN + BERT
  python code/run.py --mode train --config config_bert.py --parameters=gpus=[0],cell="lrn",layer_norm=True,output_dir="train_ln_bert" >& log.ln.bert
  ```
  Other hyperparameter settings are available in the given config.py.
  
  - Test the model as follows:
  ```
  # LRN
  python code/run.py --mode test --config config.py --parameters=gpus=[0],cell="lrn",layer_norm=False,output_dir="train_no_ln/best",test_output="out.noln" >& log.noln.test
  # LRN + LN
  python code/run.py --mode test --config config.py --parameters=gpus=[0],cell="lrn",layer_norm=True,output_dir="train_ln/best",test_output="out.ln" >& log.ln.test
  # LRN + BERT
  python code/run.py --mode test --config config_bert.py --parameters=gpus=[0],cell="lrn",layer_norm=False,output_dir="train_no_ln_bert/best",test_output="out.noln.bert" >& log.noln.bert.test
  # LRN + LN + BERT
  python code/run.py --mode test --config config_bert.py --parameters=gpus=[0],cell="lrn",layer_norm=True,output_dir="train_ln_bert/best",test_output="out.ln.bert" >& log.ln.bert.test
  ```

## Credits

Source code structure is adapted from [zero](https://github.com/bzhangGo/zero).