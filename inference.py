from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_bnb_config, set_random_seeds, read_json, dict_to_device, save_json
from preprocess import ChineseDataset, collate_func


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="ADL HW3 Instruction Tuning")
    parser.add_argument("--method", type=str,
                        default="lora-fine-tune",
                        help="support method: zero-shot, few-shot, and lora-fine-tune")
    parser.add_argument("--model_path", type=str,
                        default="",
                        help="Path to the checkpoint.")
    parser.add_argument("--adapt_checkpoint_path",
                        type=str,
                        default="",
                        help="Path to the saved adapt checkpoint.")
    parser.add_argument("--input_path", type=str,
                        default="data/public_test.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="public_prediction.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    # To ensure same seeds
    set_random_seeds()
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test_data = read_json(args.input_path)
    test_dataset = ChineseDataset(
        test_data, tokenizer, is_train=False,
        incontext=True if args.method == "few-shot" else False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)

    # Prepare model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Get bnb config
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model = PeftModel.from_pretrained(model, args.adapt_checkpoint_path)
    model.eval()

    prediction_list = []
    # Output prediction
    for _, batch_data in enumerate(test_loader, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            generated_tokens = model.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_new_tokens=512,
            )
            generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            generations = generations.replace(batch_data["prompt"][0], "").strip()
            prediction_list.append({"id": batch_data["id"][0], "output": generations})

    save_json(prediction_list, args.output_path)