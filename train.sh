# #!/bin/bash

python qlora.py --dataset data/train.json \
                --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
                --learning_rate 0.0008  \
                --per_device_eval_batch_size 128\
                --per_device_train_batch_size 256\

# Option: "yentinglin/Llama-3-Taiwan-8B-Instruct"