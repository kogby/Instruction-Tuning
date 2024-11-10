# Instruction-Tuning

This repository is the implementation of HW3 for CSIE Applied Deep Learning in 2024 Fall.

## Setting the Environment

To set the environment for inferencing, run this command:

```
pip install -r requirements.txt
```

To set the environment for Training, run this command:

```
pip install -r requirements_qlora.txt
```

## Download LoRA checkpoint

To download the checkpoint of LoRA, run the command:

```
bash ./download.sh
```

## Reproducing

To reproduce the result, run the command:

```
bash ./run.sh <pretrain model path> <lora checkpoint path> <input data path> <output file path>
```

Note: I use `zake7749/gemma-2-2b-it-chinese-kyara-dpo` for fine-tuning.

For example:

```
python inference.py --model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
                --adapt_checkpoint_path adapter_checkpoint/ \
                --input_path data/private_test.json \
                --output_path prediction.json 
```

## Training

To fine-tune the pre-trained model, run the command:

```
python qlora.py --dataset data/train.json \
				--model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
				--learning_rate 0.0008 \
				--per_device_eval_batch_size 8\
 				--per_device_train_batch_size 8\
 				--max_steps 5000\
```

You can adjust and add any parameters as long as qlora.py accepts it.

## Operating Environment

The training was conducted on Colab Pro, with A100 and 80GB RAM.
Check train_colab.ipynb for more information about the implementaion on Google Colab.

Please note that the default version of many packages on Colab is not compatible with the reruirements of qlora.py. If any problem is encountered, utilize the commented "pip install" commands.

## Reference
Qlora : https://github.com/artidoro/qlora \
zake7749/Gemma-2 :https://huggingface.co/zake7749/gemma-2-2b-it-chinese-kyara-dpo
