# #!/bin/bash

python qlora.py --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
                --learning_rate 0.0008  \
                --per_device_eval_batch_size 2\