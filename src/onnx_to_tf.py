#! /usr/bin/python3

import onnx_tf
import onnx
import tvm
import tvm.relay as relay
import numpy as np
import constants
import torch
from config import Config
from data import Dataset, BasketConstructor
from train import train_test_split
from sys import argv
from utils import sort_batch_of_lists, pad_batch_of_lists
from sys import argv
from dream import DreamModel

def main():
    dr_path = argv[1]
    dr_config = Config(constants.DREAM_CONFIG)

    with open(dr_path, 'rb') as f:
        dr_model = torch.load(f)

    bc = BasketConstructor(constants.RAW_DATA_DIR, constants.FEAT_DATA_DIR)
    ub_basket = bc.get_baskets('prior', reconstruct=False)

    # this function needs a sample input to infer types
    baskets, lens, users = Dataset(ub_basket)[0: dr_config.batch_size]

    baskets, lens, users = sort_batch_of_lists(baskets, lens, users)
    baskets = pad_batch_of_lists(baskets, lens[0])

    ub_seqs = [] # users' basket sequence

    for ubaskets in baskets:
        x = dr_model.embed_baskets(ubaskets)
        ub_seqs.append(torch.cat(x, 0).unsqueeze(0))

    ub_seqs = torch.cat(ub_seqs, 0)

    model_path = argv[2]

    onnx_model = onnx.load(model_path)
    tf_model = onnx_tf.backend.prepare(onnx_model)

    tf_model.export_graph('./models/model.pb')

if __name__ == '__main__': main()

