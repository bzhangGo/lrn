## Document Classification in LRN model

One concern of LRN is that after simplifying the recurrent component, modeling capacity, in particular the long-range
dependency, would be weakened. We answer this question by doing experiments on document classification.

We choose:
 - Amazon Review Polarity (AmaPolar, 2 labels, 3.6M/0.4M for training/testing)
 - Amazon Review Full (AmaFull, 5 labels, 3M/0.65M for training/testing)
 - Yahoo! Answers (Yahoo, 10 labels, 1.4M/60K for training/testing)
 - Yelp Review Polarity (YelpPolar, 2 labels, 0.56M/38K for training/testing)
 
Dataset comes from [Zhang et al. (2015)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf).
We use a birnn model followed by an attentive pooling layer. Char and Glove embeddings are used for word representation.
Main experimental results are summarized below.

<table>
  <tr>
    <th colspan="2" rowspan="2">Model</th>
    <th rowspan="2">#Params</th>
    <th colspan="2">AmaPolar</th>
    <th colspan="2">Yahoo</th>
    <th colspan="2">AmaFull</th>
    <th colspan="2">YelpPolar</th>
  </tr>
  <tr>
    <td>ERR</td>
    <td>Time</td>
    <td>ERR</td>
    <td>Time</td>
    <td>ERR</td>
    <td>Time</td>
    <td>ERR</td>
    <td>Time</td>
  </tr>
  <tr>
    <td colspan="2">Zhang et al. (2015)</td>
    <td>-</td>
    <td>6.10</td>
    <td>-</td>
    <td>29.16</td>
    <td>-</td>
    <td>40.57</td>
    <td>-</td>
    <td>5.26</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">This<br>Work</td>
    <td>LSTM</td>
    <td>227K</td>
    <td>4.37</td>
    <td>0.947</td>
    <td>24.62</td>
    <td>1.332</td>
    <td>37.22</td>
    <td>1.003</td>
    <td>3.58</td>
    <td>1.362</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>176K</td>
    <td>4.39</td>
    <td>0.948</td>
    <td>24.68</td>
    <td>1.242</td>
    <td>37.20</td>
    <td>0.982</td>
    <td>3.47</td>
    <td>1.230</td>
  </tr>
  <tr>
    <td>ATR</td>
    <td>74K</td>
    <td>4.78</td>
    <td>0.867</td>
    <td>25.33</td>
    <td>1.117</td>
    <td>38.54</td>
    <td>0.836</td>
    <td>4.00</td>
    <td>1.124</td>
  </tr>
  <tr>
    <td>SRU</td>
    <td>194K</td>
    <td>4.95</td>
    <td>0.919</td>
    <td>24.78</td>
    <td>1.394</td>
    <td>38.23</td>
    <td>0.907</td>
    <td>3.99</td>
    <td>1.310</td>
  </tr>
  <tr>
    <td>LRN</td>
    <td>151K</td>
    <td>4.98</td>
    <td>0.731</td>
    <td>25.07</td>
    <td>1.038</td>
    <td>38.42</td>
    <td>0.788</td>
    <td>3.98</td>
    <td>1.022</td>
  </tr>
</table>

*Time*: time in seconds per training batch measured from 1k training steps.

## Requirement
tensorflow >= 1.8.1

## How to Run?

- download and preprocess dataset

  - The dataset link: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
  - Prepare embedding and vocabulary
  
    Download the [pre-trained GloVe embedding](http://nlp.stanford.edu/data/glove.840B.300d.zip).
    Generate vocabulary for each task as follows:
    ```
    task=(amafull amapolar yahoo yelppolar)
    python code/run.py --mode vocab --config config.py --parameters=task="${task}",output_dir="${task}_vocab"
    ```
    

- training and evaluation

  - Train the model as follows:
  ```
  # configure your cuda libaray if necessary
  export CUDA_ROOT=XXX
  export PATH=$CUDA_ROOT/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

  task=(amafull amapolar yahoo yelppolar)
  python code/run.py --mode train --config config.py --parameters=task="${task}",output_dir="${task}_train",gpus=[1],word_vocab_file="${task}_vocab/vocab.word",char_vocab_file="${task}_vocab/vocab.char",enable_hierarchy=False,nthreads=2,enable_bert=False,cell="lrn",swap_memory=False
  ```
  Other hyperparameter settings are available in the given config.py.
  
  - Test the model as follows:
  ```
  task=(amafull amapolar yahoo yelppolar)
  python code/run.py --mode test --config config.py --parameters=task="${task}",output_dir="${task}_train/best",gpus=[0],word_vocab_file="${task}_vocab/vocab.word",char_vocab_file="${task}_vocab/vocab.char",enable_hierarchy=False,nthreads=2,enable_bert=False,cell="lrn",swap_memory=False,train_continue=False,test_output=${task}.out.txt
  ```

## Credits

Source code structure is adapted from [zero](https://github.com/bzhangGo/zero).