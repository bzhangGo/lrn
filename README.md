# lrn
Source code for "A Lightweight Recurrent Network for Sequence Modeling"


## Model Architecture
In our new paper, we propose lightweight recurrent network, which combines the strengths of 
[ATR](https://arxiv.org/abs/1810.12546) and [SRU](https://arxiv.org/abs/1709.02755). 

* ATR helps reduces model parameters and avoids additional free parameters for gate calculation, through the twin-gate
mechanism
* SRU follows the [QRNN](https://arxiv.org/abs/1611.01576) and moves all recurrent computations outside the recurrence.

Based on the above units, we propose [LRN](xxx):

<img src="https://latex.codecogs.com/svg.latex?\begin{align*}&space;\mathbf{q}_t,&space;\mathbf{k}_t,&space;\mathbf{v}_t&space;=&space;\mathbf{x}_t\mathbf{W}_q,&space;\mathbf{x}_t\mathbf{W}_k,&space;\mathbf{x}_t\mathbf{W}_v&space;\\&space;\mathbf{i}_t&space;=&space;\sigma(\mathbf{k}_t&space;&plus;&space;\mathbf{h}_{t-1})&space;\\&space;\mathbf{f}_t&space;=&space;\sigma(\mathbf{q}_t&space;-&space;\mathbf{h}_{t-1})&space;\\&space;\mathbf{h}_t&space;=&space;g(\mathbf{i}_t&space;\odot&space;\mathbf{v}_t&space;&plus;&space;\mathbf{f}_t&space;\odot&space;\mathbf{h}_{t-1})&space;\end{align}" title="\begin{align} \mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t = \mathbf{x}_t\mathbf{W}_q, \mathbf{x}_t\mathbf{W}_k, \mathbf{x}_t\mathbf{W}_v \\ \mathbf{i}_t = \sigma(\mathbf{k}_t + \mathbf{h}_{t-1}) \\ \mathbf{f}_t = \sigma(\mathbf{q}_t - \mathbf{h}_{t-1}) \\ \mathbf{h}_t = g(\mathbf{i}_t \odot \mathbf{v}_t + \mathbf{f}_t \odot \mathbf{h}_{t-1}) \end{align*}"/>

where g(&middot;) is an activation function, *tanh* or *identity*. W<sub>q</sub>, W<sub>k</sub> and W<sub>v</sub> 
are model parameters. The matrix computation (as well as potential layer noramlization) can be shfited outside the 
recurrence. Therefore, the whole model is fast in running.

When applying twin-gate mechanism, the output value in **h**<sub>t</sub> might suffer explosion issue, 
which could grow into infinity. This is the reason we added the activation function. Another alternative solution
would be using layer normalization, which forces activation values to be stable.

## Structure Analysis
One way to understand the model is to unfold the LRN structure along input tokens:

<img src="https://latex.codecogs.com/svg.latex?\mathbf{h}_t&space;&&space;=&space;\sum_{k=1}^t&space;\mathbf{i}_k&space;\odot&space;\left(\prod_{l=1}^{t-k}\mathbf{f}_{k&plus;l}\right)&space;\odot&space;\mathbf{v}_k," title="\mathbf{h}_t & = \sum_{k=1}^t \mathbf{i}_k \odot \left(\prod_{l=1}^{t-k}\mathbf{f}_{k+l}\right) \odot \mathbf{v}_k,"/>

The above structure which is also observed by [Zhang et al.](https://arxiv.org/abs/1810.12546), [Lee et al.](https://arxiv.org/abs/1705.07393), 
and etc, endows the RNN model with multiple interpretations. We provide two as follows:

* *Relation with Self Attention Networks*
<img src="figures/san_corr.png" width=300 />

Informally, LRN assembles forget gates from step *t* to step *k+1* in order to query the key (input gate). The result 
weight is assigned to the corresponding value representation and contributes to the final hidden representation.

Does the learned weights make sense? We do a classification tasks on AmaPolar task with a unidirectional linear-LRN.
The final hidden state is feed into the classifier. One example below shows the learned weights. The term *great* gains
a large weight, which decays slowly and contributes the final *positive* decision.
<img src="figures/memory.png"  width=500 />

* *Long-term and Short-term Memory*
<img src="figures/ls_mem.png"  width=250 />

Another view of the unfolded structure is that different gates form different memory mechanism. The input gate acts as
a short-term memory and indicates how many information can be activated in this token. The forget gates form a forget
chain that controls how to erase meaningless past information.

## Experiments 

We did experiment on six different tasks:
* [Natural Language Inference](nli)
* [Document Classification](doc)
* [Machine Translation](mt)
* [Reading Comprehension](rc)
* [Named Entity Recognition](ner)
* [Language Modeling](lm)


## Citation

Please cite the following paper:
> Biao Zhang; Rico Sennrich (2019). *A Lightweight Recurrent Network for Sequence Modeling*. 
In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Florence, Italy.

```
@inproceedings{zhang-sennrich:2019:ACL,
  address = "Florence, Italy",
  author = "Zhang, Biao and Sennrich, Rico",
  booktitle = "{Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics}",
  publisher = "Association for Computational Linguistics",
  title = "{A Lightweight Recurrent Network for Sequence Modeling}",
  year = "2019"
}
```

## Contact

For any further comments or questions about LRN, please email <a href="mailto:b.zhang@ed.ac.uk">Biao Zhang</a>.