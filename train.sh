# #!/bin/bash

python train.py --base_model_path "zake7749/gemma-2-2b-it-chinese-kyara-dpo" \
                --train_data_path data/train.json \
                --valid_data_path data/public_test.json \
                --train_num 10000 \
                --epoch 7 \
                --batch_size 512 \
                --accum_grad_step 8 \
                --lr <learning rate> \
                --lr_scheduler "cosine" \
                --warm_up_step 300 \
                --lora_rank 64