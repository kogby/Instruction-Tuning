#!/bin/bash

python inference.py --model_path "${1}" \
                --adapt_checkpoint_path "${2}" \
                --input_path "${3}" \
                --output_path "${4}"

# python inference.py --model_path "${1}" \
#                 --adapt_checkpoint_path "${2}" \
#                 --input_path "${3}" \
#                 --output_path "${4}"
#                 --method few-shot

# Option: "yentinglin/Llama-3-Taiwan-8B-Instruct"