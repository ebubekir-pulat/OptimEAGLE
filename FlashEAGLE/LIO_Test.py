print("\n\n*******************************\nStarting LIO_Test.py\n\n")

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import Compress
import hashlib

dataset = "THUDM/LongBench"
LIO_model_paths = ["deepseek-ai/DeepSeek-R1"]
base_model_paths = ["Qwen/Qwen3-1.7B"]
EAGLE_model_paths = ["AngelSlim/Qwen3-1.7B_eagle3"]
# Note: Reference for Qwen3: https://huggingface.co/Qwen/Qwen3-1.7B, https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3

if dataset == "THUDM/LongBench":
    prompts = Data.longbench_e()
elif dataset == "PKU-Alignment/Align-Anything-Instruction-100K-zh":
    prompts = Data.aai_dataset()


LIO_prompt = f"Generate optimal hyperparameters for EAGLE-3 speculative decoding with SGLANG, where the \
            hyperparameters are --speculative-num-steps, --speculative-eagle-topk, --speculative-num-draft-tokens and --cuda-graph-max-bs. \
            The base model to be used is {base_model_paths[0]}, the EAGLE-3 model to be used is {EAGLE_model_paths[0]} \
            and the dataset to be tested on is {dataset}. \
            \nGenerate the hyperparameters in the format: \
            \nspeculative-num-steps: *value* #1 \
            \nspeculative-eagle-topk: *value* #2 \
            \nspeculative-num-draft-tokens: *value* #3 \
            \ncuda-graph-max-bs: *value* #4"


# Preparing LIO SGLANG
# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html, https://docs.sglang.ai/basic_usage/send_request.html
server_process, port = launch_server_cmd(
    f"""
python3 -m sglang.launch_server --model-path {LIO_model_paths[0]} --dtype float16
"""
)

# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
response = client.chat.completions.create(
    model=LIO_model_paths[0],
    messages=[
        {"role": "user", "content": LIO_prompt},
    ],
    temperature=0.0,
    max_tokens=2048,
)

# Reference for below code line: https://stackoverflow.com/questions/77444332/openai-python-package-error-chatcompletion-object-is-not-subscriptable 
LIO_output = response.choices[0].message.content

# Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
terminate_process(server_process)

num_steps, topk, draft_tokens, graph_max_bs = Data.extract_LIO_response(LIO_output)

# Preparing SGLANG with EAGLE3
# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
server_process, port = launch_server_cmd(
    f"""
python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path {EAGLE_model_paths[0]} --speculative-num-steps {num_steps} \
        --speculative-eagle-topk {topk} --speculative-num-draft-tokens {draft_tokens} --mem-fraction 0.6 \
        --cuda-graph-max-bs {graph_max_bs} --dtype float16
"""
)

# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


LIO_outputs = []
summarise = True
ranked_retrieve = False
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Summarise: ", summarise, "\n")
print("Ranked Retrieve: ", ranked_retrieve)

# LongBench-E Assessment Loop
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
        if summarise == True:
            prompt = Compress.summarise_question(prompts[i][0] + "\n" + prompts[i][1])
        elif ranked_retrieve == True:
            prompt = Compress.ranked_retrieve(prompts[i][0], prompts[i][1]) + "\n" + prompts[i][1]
        
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
        model_output = response.choices[0].message.content
        
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
            "id": hashlib.md5((prompt + model_output).encode()).hexdigest(),
            "output": model_output
        }
        LIO_outputs.append(output)

# Print LongBench-E Results
print(f"LIO Results for {LIO_model_paths[0]}:")
print(f"Dataset: {dataset}")
print(f"EAGLE Model: {EAGLE_model_paths[0]}")
print(f"Base Model: {base_model_paths[0]}")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens Generated/s: ", np.mean(token_rates))
#print("Average Acceptance Length: ", np.mean(avg_accept_lens))

# Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
terminate_process(server_process)

compression_tag = ""
if summarise == True:
    compression_tag = "_Summ"
elif ranked_retrieve == True:
    compression_tag == "_RR"

output_name = f"LIO_Output_{LIO_model_paths[0]}_{EAGLE_model_paths[0]}{compression_tag}.jsonl" 
 
# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(output_name, "x") as f:
    for output in LIO_outputs:
        f.write(json.dumps(output) + "\n")


print("\n\n*******************************\nFinished Running LIO_Test.py\n\n")

''' 
References
1.
'''