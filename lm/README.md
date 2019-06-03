## Language Modeling


We do experiments on PTB and WT2 dataset, and use the mixture of softmax model [MoS](https://arxiv.org/abs/1711.03953).
Main experimental results are summarized below.

<table>
  <tr>
    <th colspan="2" rowspan="2">Model</th>
    <th rowspan="2">#Params</th>
    <th colspan="3">PTB</th>
    <th colspan="3">WT2</th>
  </tr>
  <tr>
    <td>Base</td>
    <td>+Finetune</td>
    <td>+Dynamic</td>
    <td>Base</td>
    <td>+Finetune</td>
    <td>+Dynamic</td>
  </tr>
  <tr>
    <td colspan="2">Yang et al. (2018)</td>
    <td>22M</td>
    <td>55.97</td>
    <td>54.44</td>
    <td>47.69</td>
    <td>63.33</td>
    <td>61.45</td>
    <td>40.68</td>
  </tr>
  <tr>
    <td rowspan="5">This<br>Work</td>
    <td>LSTM</td>
    <td>22M</td>
    <td>63.78</td>
    <td>62.12</td>
    <td>53.11</td>
    <td>69.78</td>
    <td>68.68</td>
    <td>44.60</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>17M</td>
    <td>69.09</td>
    <td>67.61</td>
    <td>60.21</td>
    <td>73.37</td>
    <td>73.05</td>
    <td>49.77</td>
  </tr>
  <tr>
    <td>ATR</td>
    <td>9M</td>
    <td>66.24</td>
    <td>65.86</td>
    <td>58.29</td>
    <td>75.36</td>
    <td>73.35</td>
    <td>48.65</td>
  </tr>
  <tr>
    <td>SRU</td>
    <td>13M</td>
    <td>69.64</td>
    <td>65.29</td>
    <td>60.97</td>
    <td>85.15</td>
    <td>84.97</td>
    <td>57.97</td>
  </tr>
  <tr>
    <td>LRN</td>
    <td>11M</td>
    <td>61.26</td>
    <td>61.00</td>
    <td>54.45</td>
    <td>69.91</td>
    <td>68.86</td>
    <td>46.97</td>
  </tr>
</table>

Test perplexity.

## Requirement
PyTorch >= 0.4.1

## How to Run?
- download and preprocess dataset

  - see [MoS](https://github.com/zihangdai/mos) about the preprocessing of datasets

- training and evaluation

  - training
  ```
  #! /bin/bash

  export CUDA_VISIBLE_DEVICES=0

  # for PTB
  python3 main.py --data path-of/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 10.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB --single_gpu --model lrn
  # for WT2
  python3 main.py --epochs 1000 --data path-of/wikitext-2 --save WT2 --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --nhidlast 650 --emsize 300 --batch_size 15 --lr 15.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu --model lrn  
  ```
  
  - finetuning
  ```
  # for PTB
  python3 finetune.py --data path-of/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 15.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15 --save PTB-XXX --single_gpu --model lrn
  # for WT2
  python3 finetune.py --epochs 1000 --data path-of/wikitext-2 --save WT2-XXX --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --emsize 300 --batch_size 15 --lr 20.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu --model lrn
  ```
  
  - dynamic evaluation
  ```
  # for PTB
  python3 dynamiceval.py --model PTB-XXX/finetune_model.pt --data path-of/penn --lamb 0.075 --gpu 0
  # for WT2
  python3 dynamiceval.py --data path-of/wikitext-2 --model WT2-XXX/finetune_model.pt --epsilon 0.002 --gpu 0
  ```
  
  - general evaluation
  ```
  # for PTB
  python3 evaluate.py --data path-of/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 10.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB-XXX --single_gpu --model lrn
  # for WT2
  python3 evaluate.py --epochs 1000 --data path-of/wikitext-2 --save WT2-XXX --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --nhidlast 650 --emsize 300 --batch_size 15 --lr 15.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu --model lrn
  ```

## Credits

Source code structure is adapted from [MoS](https://github.com/zihangdai/mos).