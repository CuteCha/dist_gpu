#!/bin/bash

~/py_env/tf2/bin/saved_model_cli show --dir output/model/house/saved_graph --tag_set serve --signature_def serving_default
~/py_env/tf2/bin/saved_model_cli show --dir output/model/house/saved_graph --tag_set serve --signature_def serving_default
 ~/py_env/tf2/bin/saved_model_cli run --dir output/model/house/saved_graph --tag_set serve --signature_def serving_default --input_exprs 'dense_input=np.ones((2,8))'