#! /usr/bin/python

import sys
import numpy as np


def extract_dev_test_score(fname):
    test_score = float(open(fname, 'rU').readlines()[-1].strip())

    return test_score


cell_type = sys.argv[1]
exp_dirs = sys.argv[2:]

scores = []
for exp_dir in exp_dirs:
    test_score = extract_dev_test_score("{}/log.{}".format(exp_dir, cell_type))
    scores.append(test_score)

print(np.mean(scores), np.std(scores))
