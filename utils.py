from transformers import BitsAndBytesConfig
import torch
import json
import random
import numpy as np

def get_prompt(instruction: str, incontext: bool = True) -> str:
    '''Format the instruction as a prompt for LLM.'''
    if incontext:
        return f"""你是精通古今中文的翻譯助理，以下是用戶和翻譯助理之間的對話。
             你要對用戶的問題提供詳細、精準的回答。將文言文翻譯成白話文，或白話文翻譯成文言文。\
             這邊提供你兩個範例。
             USER:翻譯成文言文：雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。
             ASSISTANT:雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。
             USER: 能服信政，此謂正紀。翻譯成現代文：
             ASSISTANT: 能守信於民，這叫作端正綱紀。
             USER: {instruction} ASSISTANT:"""
    else:
        return f"你是中文能力極高的助理，以下是用戶和助理之間的對話。你要對用戶的問題提供有用、詳細並且精準翻譯的回答。以下的問題為文言文翻譯成白話文或白話文翻譯成文言文，請回答：USER: {instruction} ASSISTANT:"


def get_bnb_config() -> BitsAndBytesConfig:
    # Ref: https://huggingface.co/blog/zh/4bit-transformers-bitsandbytes
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def set_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4, ensure_ascii=False)
    return


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}