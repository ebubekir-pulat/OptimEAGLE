print("\n\n*******************************\nStarting EAGLE3_Lossy.py\n\n")

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import hashlib

base_model_paths = ["Qwen/Qwen3-1.7B"]
EAGLE_model_paths = ["AngelSlim/Qwen3-1.7B_eagle3"]
# Note: Reference for Qwen3: https://huggingface.co/Qwen/Qwen3-1.7B, https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3

dataset = "THUDM/LongBench"
if dataset == "THUDM/LongBench":
    prompts = Data.longbench_e()
elif dataset == "PKU-Alignment/Align-Anything-Instruction-100K-zh":
    prompts = Data.aai_dataset()

accept_threshold = 0.9
threshold_acc = 1.0

# Preparing SGLANG with EAGLE3
# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
server_process, port = launch_server_cmd(
    f"""
python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path {EAGLE_model_paths[0]} --speculative-accept-threshold-single {accept_threshold} \
        --speculative-accept-threshold-acc {threshold_acc} --mem-fraction 0.6 --cuda-graph-max-bs 2 --dtype float16
"""
)

# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

lossy_outputs = []
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Dataset: ", dataset)
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Speculative Accept Threshold Single: ", accept_threshold)
print("Speculative Accept Threshold Acc: ", threshold_acc)

# Lossy EAGLE3 Assessment Loop
wall_times = []
token_rates = []
input_tokens = []
output_tokens = []

for test_run in range(test_runs):
    run = 1
    for i in range(len(prompts)):
        print("Test Run: ", test_run)
        print("Test Question: ", run)
        run += 1

        prompt = prompts[i][0] + "\n" + prompts[i][1]
        
        start = time.perf_counter_ns()

        # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
        response = client.chat.completions.create(
            model=base_model_paths[0],
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=max_new_tokens,
        )

        finish = time.perf_counter_ns()

        # Reference for below code line: https://stackoverflow.com/questions/77444332/openai-python-package-error-chatcompletion-object-is-not-subscriptable 
        lossy_output = response.choices[0].message.content
        
        elapsed = finish - start
        wall_times.append(elapsed)

        new_tokens = response.usage.completion_tokens
        tokens_per_second = new_tokens / (elapsed * pow(10, -9))
        token_rates.append(tokens_per_second)
        output_tokens.append(new_tokens)

        input_tokens = response.usage.prompt_tokens

        # Reference for below code block: https://github.com/SafeAILab/EAGLE/issues/153
        #steps = int(output_ids[2])
        #avg_accept_len = new_tokens / steps
        #avg_accept_lens.append(avg_accept_len)

        # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
        output = {
            "id": hashlib.md5((prompt + lossy_output).encode()).hexdigest(),
            "output": lossy_output
        }
        lossy_outputs.append(output)

# Print LossyEAGLE-3 Results
print(f"LossyEAGLE-3 for {EAGLE_model_paths[0]}:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens Generated/s: ", np.mean(token_rates))
#print("Average Acceptance Length: ", np.mean(avg_accept_lens))

# Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
terminate_process(server_process)

output_name = f"LossyEAGLE3_Output_{EAGLE_model_paths[0].replace("/", "-")}_{accept_threshold}_{threshold_acc}.jsonl" 

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(output_name, "x") as f:
    for output in lossy_outputs:
        f.write(json.dumps(output) + "\n")


print("\n\n*******************************\nFinished Running EAGLE3_Lossy.py\n\n")

''' 
References
1.
'''