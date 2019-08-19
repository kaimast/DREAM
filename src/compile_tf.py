#! /usr/bin/python3

import tvm.relay.testing.tf as tf_testing
import tvm
import nnvm
import tensorflow as tf
import tvm.relay as relay

from sys import argv

def main():
    # target setttings
    target = 'llvm'
    target_host = 'llvm'
    layout = None
    ctx = tvm.cpu(0)

    model_path = argv[1]
    graph_def = tf.GraphDef()

    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    shape_dict = {}
    
    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    print("Tensorflow protobuf imported to relay frontend.")

    with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

if __name__ == '__main__': main()

