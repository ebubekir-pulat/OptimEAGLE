print("\n\n*******************************\nStarting AAI_Test.py\n\n")

import time
import numpy as np
import torch
import json
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import Data
import hashlib
import Compress
from threading import Lock
import threading

input_buffer = []
output_buffer = []
input_lock = Lock()
output_lock = Lock()
finish_outputs = [0]

original_aai_ds = Data.aai_dataset()
num_prompts = len(original_aai_ds)
aai_ds = []
aai_outputs = []

def prepare_inputs():
    while len(original_aai_ds) > 0:
        aai_prompt = original_aai_ds.pop(0)
        aai_prompt = Compress.zh_to_en(aai_prompt)[0]
        aai_ds.append(aai_prompt)

def process_outputs(translate):
    curr = 0

    while curr <= num_prompts:
        if len(aai_outputs) > curr:
            aai_output = aai_outputs[curr]
            if translate == True:
                # Below Code Line From: https://github.com/SafeAILab/EAGLE
                aai_output = Compress.en_to_zh(model.tokenizer.decode(aai_output[0][0]))[0]
            else:
                aai_output = model.tokenizer.decode(aai_output[0][0])
            
            aai_outputs[curr] = aai_output
    
    finish_outputs[0] = 1

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


models_to_test = [0]
translate = True
test_runs = 1
max_new_tokens = 128
temp = 0.0

if translate == False:
    aai_ds = original_aai_ds
else:
    # Reference for below code block: https://www.geeksforgeeks.org/python/multithreading-python-set-1/
    inputs_thread = threading.Thread(prepare_inputs)
    inputs_thread.start()
    outputs_thread = threading.Thread(process_outputs, args=[translate])
    outputs_thread.start()

print("\nEvaluation Settings Chosen:")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Translate: ", translate, "\n")

new_tokens_total = 0
start = time.perf_counter_ns()

# AAI Dataset Assessment Loop
for model_index in models_to_test:
    #wall_times = []
    #token_rates = []
    avg_accept_lens = []
    model = model_init(model_index)
    for test_run in range(test_runs):
        run = 1
        for i in range(num_prompts):
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            while len(aai_ds) == i:
                print("", end="")
            your_message = aai_ds[i]

            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            conv = get_conversation_template(template_getter(model_index))
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            output_ids = model.eagenerate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, log=True)
            aai_outputs.append(output_ids)
            
            new_tokens_total += int(output_ids[1])

            # Reference for below code block: https://github.com/SafeAILab/EAGLE/issues/153
            steps = int(output_ids[2])
            avg_accept_len = int(output_ids[1]) / steps
            avg_accept_lens.append(avg_accept_len)

while finish_outputs[0] == 0:
    print("", end="")

finish = time.perf_counter_ns()
elapsed = finish - start
#wall_times.append(elapsed)

tokens_per_second = new_tokens_total / (elapsed * pow(10, -9))

# Print AAI Dataset Results
print(f"AAI Results for {EAGLE_model_paths[model_index]}:")
print("Wall Time (ns): ", elapsed)
print("Mean Tokens Generated/s: ", tokens_per_second)
print("Average Acceptance Length: ", np.mean(avg_accept_lens))


AAI_records = []

for i in range(len(aai_outputs)):
    # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
    output = {
        "id": hashlib.md5((aai_ds[i] + Data.extract_response(aai_outputs[i])).encode()).hexdigest(),
        "output": Data.extract_response(aai_outputs[i])
    }
    AAI_records.append(output)

translate_tag = ""

if translate == True:
    translate_tag = "_Translate"

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(f"AAI_Output{translate_tag}_{EAGLE_model_paths[model_index]}.jsonl", "x") as f:
    for output in AAI_records:
        f.write(json.dumps(output) + "\n")

print("\n\n*******************************\nFinished Running AAI_Test.py\n\n")