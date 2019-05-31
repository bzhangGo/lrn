## Machine Translation in LRN model


Main source code will be available at [zero](https://github.com/bzhangGo/zero) (might require some time, 31/05/2019).
The used NMT structure is in `deepnmt.py`.


Main experimental results are summarized below.

<table>
  <tr>
    <th>Model</th>
    <th>#Params</th>
    <th>BLEU</th>
    <th>Train</th>
    <th>Decode</th>
  </tr>
  <tr>
    <td>GNMT</td>
    <td>-</td>
    <td>24.61</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>206M</td>
    <td>26.28</td>
    <td>2.67</td>
    <td>45.35</td>
  </tr>
  <tr>
    <td>ATR</td>
    <td>122M</td>
    <td>25.70</td>
    <td>1.33</td>
    <td>34.40</td>
  </tr>
  <tr>
    <td>SRU</td>
    <td>170M</td>
    <td>25.91</td>
    <td>1.34</td>
    <td>42.84</td>
  </tr>
  <tr>
    <td>LRN</td>
    <td>143M</td>
    <td>26.26</td>
    <td>0.99</td>
    <td>36.50</td>
  </tr>
  <tr>
    <td>oLRN</td>
    <td>164M</td>
    <td>26.73</td>
    <td>1.15</td>
    <td>40.19</td>
  </tr>
</table>

*Train*: time in seconds per training batch measured from 0.2k training steps. 
*Decode*: time in milliseconds used to decode one sentence measured on newstest2014 dataset.
*BLEU*: case-insensitive tokenized BLEU score on WMT14 English-German translation task.

## oLRN structure

<img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;\mathbf{q}_t,&space;\mathbf{k}_t,&space;\mathbf{v}_t,&space;\mathbf{x}_o&space;=&space;\mathbf{x}_t\mathbf{W}_q,&space;\mathbf{x}_t\mathbf{W}_k,&space;\mathbf{x}_t\mathbf{W}_v,&space;\mathbf{x}_t&space;\mathbf{W}_o&space;\\&space;\mathbf{i}_t&space;=&space;\sigma(\mathbf{k}_t&space;&plus;&space;\mathbf{h}_{t-1})&space;\\&space;\mathbf{f}_t&space;=&space;\sigma(\mathbf{q}_t&space;-&space;\mathbf{h}_{t-1})&space;\\&space;\mathbf{c}_t&space;=&space;g(\mathbf{i}_t&space;\odot&space;\mathbf{v}_t&space;&plus;&space;\mathbf{f}_t&space;\odot&space;\mathbf{h}_{t-1})&space;\\&space;\mathbf{o}_t&space;=&space;\sigma(\mathbf{x}_o&space;-&space;\mathbf{c}_t)&space;\\&space;\mathbf{h}_t&space;=&space;\mathbf{o}_t&space;\odot&space;\mathbf{c}_t&space;\end{align*}" title="\begin{align*} \mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t, \mathbf{x}_o = \mathbf{x}_t\mathbf{W}_q, \mathbf{x}_t\mathbf{W}_k, \mathbf{x}_t\mathbf{W}_v, \mathbf{x}_t \mathbf{W}_o \\ \mathbf{i}_t = \sigma(\mathbf{k}_t + \mathbf{h}_{t-1}) \\ \mathbf{f}_t = \sigma(\mathbf{q}_t - \mathbf{h}_{t-1}) \\ \mathbf{c}_t = g(\mathbf{i}_t \odot \mathbf{v}_t + \mathbf{f}_t \odot \mathbf{h}_{t-1}) \\ \mathbf{o}_t = \sigma(\mathbf{x}_o - \mathbf{c}_t) \\ \mathbf{h}_t = \mathbf{o}_t \odot \mathbf{c}_t \end{align*}" />

Unlike LRN, oLRN employs an additional output gate, inspired by LSTM, to handle output information flow. 
This additional gate also help avoid hidden state explosion when linear activation is applied.

## How to Run?

Training and evaluation, please refer to project [zero](https://github.com/bzhangGo/zero).