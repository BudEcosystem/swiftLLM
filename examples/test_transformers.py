import time
import torch
import numpy as np
import random
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # for reproducibility
    torch.backends.cudnn.benchmark = False


set_seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm model executor directly for inferencing without using the engine
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        required=True
    )
    model_path = parser.parse_args().model_path

    start_time = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_size="left")
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Life blooms like a flower, far away",
        "one two three four five",
        "A B C D E F G H I J K L M N O P Q R S T U V",
        "To be or not to be,",
    ]

    model.generation_config.pad_token = tokenizer.pad_token
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    print(model.generation_config)

    for i, prompt in enumerate(prompts):
        input = tokenizer(prompts[i], padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        # output = model.generate(input["input_ids"], attention_mask=input["attention_mask"], max_new_tokens=20, do_sample=False).tolist()[0]
        output = model.generate(input["input_ids"], attention_mask=input["attention_mask"], max_new_tokens=20, temperature=0.5, top_k=1, top_p=0.5).tolist()[0]
        output_text = tokenizer.decode(output[input["input_ids"][0].shape[-1]:], skip_special_tokens=True)

        print(f"{prompt}|{output_text}")