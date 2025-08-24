print("\n\n*******************************\nStarting AAI_Test.py\n\n")

import time
import numpy as np
import torch
import json
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import Data
import hashlib

base_model_paths = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
EAGLE_model_paths = ["yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"]

def template_getter(model_index):
    return base_model_paths[model_index]

def model_init(model_index):
    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[model_index],
        ea_model_path=EAGLE_model_paths[model_index],
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model


aai_ds = Data.aai_dataset()
AAI_outputs = []
models_to_test = [0]
translate = True
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Translate: ", translate, "\n")

# AAI Dataset Assessment Loop
for model_index in models_to_test:
    wall_times = []
    token_rates = []
    avg_accept_lens = []
    model = model_init(model_index)
    for test_run in range(test_runs):
        run = 1
        for question in aai_ds:
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            start = time.perf_counter_ns()

            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            your_message = question
            if translate == True:
                your_message = Data.zh_to_en(your_message)
            conv = get_conversation_template(template_getter(model_index))
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            output_ids = model.eagenerate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, log=True)

            if translate == True:
                # Below Code Line From: https://github.com/SafeAILab/EAGLE
                aai_output = Data.en_to_zh(model.tokenizer.decode(output_ids[0][0]))
            else:
                # Below Code Line From: https://github.com/SafeAILab/EAGLE
                aai_output = model.tokenizer.decode(output_ids[0][0])

            finish = time.perf_counter_ns()
            elapsed = finish - start
            wall_times.append(elapsed)

            new_tokens = int(output_ids[1])
            tokens_per_second = new_tokens / (elapsed * pow(10, -9))
            token_rates.append(tokens_per_second)

            # Reference for below code block: https://github.com/SafeAILab/EAGLE/issues/153
            steps = int(output_ids[2])
            avg_accept_len = new_tokens / steps
            avg_accept_lens.append(avg_accept_len)

            # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
            output = {
                "id": hashlib.md5((question + aai_output).encode()).hexdigest(),
                "output": aai_output
            }
            AAI_outputs.append(output)


    # Print AAI Dataset Results
    print(f"AAI Results for {base_model_paths[model_index]}:")
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))
    print("Average Acceptance Length: ", np.mean(avg_accept_lens))


# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(f"AAI_output_{translate}_{base_model_paths[model_index]}.jsonl", "w") as f:
    for output in AAI_outputs:
        f.write(json.dumps(output) + "\n")

print("\n\n*******************************\nFinished Running AAI_Test.py\n\n")