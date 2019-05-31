#! /usr/bin/python

import sys

data = open(sys.argv[1], 'r')
data.readline()

out_l = open(sys.argv[2]+".l", 'w')
out_p = open(sys.argv[2]+".p", 'w')
out_q = open(sys.argv[2]+".q", 'w')

label = {'entailment': 0,
         'neutral': 1,
         'contradiction': 2}

for line in data:
    l, p, q = line.strip().split('\t')[:3]
    if l not in label:
        continue
    out_l.write(str(label[l]) + '\n')
    out_p.write(p.replace('( ', '').replace(' )', '') + '\n')
    out_q.write(q.replace('( ', '').replace(' )', '') + '\n')

out_l.close()
out_p.close()
out_q.close()
