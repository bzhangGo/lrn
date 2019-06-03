#! /bin/bash

export CUDA_ROOT=XXX
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

export name=log_lrn

python config.py --mode test --cell lrn --batch_size 8

