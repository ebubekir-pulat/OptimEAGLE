print("\n\n*******************************\nStarting Longbench_E_Test.py\n\n")

import time
import numpy as np
import torch
import json
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template

base_model_paths = ["lmsys/vicuna-13b-v1.3",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.3-70B-Instruct",
                    "Qwen/Qwen3-1.7B"]

EAGLE_model_paths = ["yuhuili/EAGLE3-Vicuna1.3-13B",
                     "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
                     "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                     "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
                     "AngelSlim/Qwen3-1.7B_eagle3"]

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




LB_outputs = []
models_to_test = [4]
summarise = True

print("\nEvaluation Settings Chosen:")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Summarise: ", summarise, "\n")

# LongBench-E Assessment Loop
for model_index in models_to_test:
    wall_times = []
    token_rates = []
    avg_accept_lens = []
    model = model_init(model_index)
    for test_run in range(test_runs):
        run = 1
        for i in range(len(lb_prompts)):
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            start = time.perf_counter_ns()

            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            your_message = lb_prompts[i][0] + "\n\n" + lb_prompts[i][1]
            if summarise == True:
                your_message = summarise_question(lb_prompts[i][0]) + "\n\n" + lb_prompts[i][1]
            conv = get_conversation_template(template_getter(model_index))
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            output_ids = model.eagenerate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, log=True)

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            lb_output = model.tokenizer.decode(output_ids[0])

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
                "id": run,
                "output": lb_output
            }
            LB_outputs.append(output)

    # Print LongBench-E Results
    print(f"LongBench-E Results for {base_model_paths[model_index]}:")
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))
    print("Average Acceptance Length: ", np.mean(avg_accept_lens))

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open("LBE_output.jsonl", "w") as f:
    for output in LB_outputs:
        f.write(json.dumps(output) + "\n")