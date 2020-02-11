#!/bin/sh
export CUDA_VISIBLE_DEVICES=""
tensorboard --logdir runs --port 6006 --bind_all