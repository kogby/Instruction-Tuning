# #!/bin/bash

python qlora.py --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
                --learning_rate 0.0008  \
                --per_device_eval_batch_size 2\

# python train.py --base_model_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" \
#                 --train_data_path data/train.json \
#                 --valid_data_path data/public_test.json \
#                 --train_num 10000 \
#                 --epoch 7 \
#                 --batch_size <number of training batch size> \
#                 --accum_grad_step <number of accumulated gradient batch size> \
#                 --lr <learning rate> \
#                 --lr_scheduler "cosine" \
#                 --warm_up_step 300 \
#                 --lora_rank 64