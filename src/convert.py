#! /usr/bin/python3

import constants
import torch
from config import Config
from dream import DreamModel
import torch_tvm
from data import Dataset, BasketConstructor
from train import train_test_split
from sys import argv
from utils import sort_batch_of_lists, pad_batch_of_lists

def main():
    torch_tvm.enable(
        opt_level=3,
        device_type="gpu",
        device="cuda",
        host="llvm")

    path = argv[1]
    dr_config = Config(constants.DREAM_CONFIG)
    with open(path, 'rb') as f:
        dr_model = torch.load(f)

    bc = BasketConstructor(constants.RAW_DATA_DIR, constants.FEAT_DATA_DIR)
    ub_basket = bc.get_baskets('prior', reconstruct=False)

    # this function needs a sample input to infer types
    baskets, lens, users = Dataset(ub_basket)[0: dr_config.batch_size]

    baskets, lens, users = sort_batch_of_lists(baskets, lens, users)
    baskets = pad_batch_of_lists(baskets, lens[0])

    dr_hidden = dr_model.init_hidden(dr_config.batch_size)
 
    ub_seqs = [] # users' basket sequence

    for ubaskets in baskets:
        x = dr_model.embed_baskets(ubaskets)
        ub_seqs.append(torch.cat(x, 0).unsqueeze(0))

    ub_seqs = torch.cat(ub_seqs, 0)

    arg = [ub_seqs, dr_hidden]

    relay_graph = torch_tvm.to_relay(dr_model.rnn, arg)

if __name__ == '__main__': main()

