print("\n\n*******************************\nStarting Longbench_E_Test.py\n\n")

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import Compress
import hashlib
from matplotlib import pyplot as plt

base_model_paths = ["Qwen/Qwen3-1.7B"]
EAGLE_model_paths = ["AngelSlim/Qwen3-1.7B_eagle3"]
# Note: Reference for Qwen3: https://huggingface.co/Qwen/Qwen3-1.7B, https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3

lb_prompts = Data.longbench_e()
eagle3 = True

if eagle3 == True:
    # Preparing SGLANG with EAGLE3
    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    server_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path {EAGLE_model_paths[0]} --speculative-num-steps 5 \
            --speculative-eagle-topk 8 --speculative-num-draft-tokens 32 --mem-fraction 0.6 \
            --cuda-graph-max-bs 2 --dtype float16
    """
    )
else:
    # Preparing AutoReg SGLANG
    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html, https://docs.sglang.ai/basic_usage/send_request.html
    server_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {base_model_paths[0]} --mem-fraction 0.6 \
            --cuda-graph-max-bs 2 --dtype float16
    """
    )


# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


LB_outputs = []
summarise = True
ranked_retrieve = False
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("EAGLE3: ", eagle3)
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
    for i in range(len(lb_prompts)):
        print("Test Run: ", test_run)
        print("Test Question: ", run)
        run += 1

        prompt = lb_prompts[i][0] + "\n" + lb_prompts[i][1]
        if summarise == True:
            prompt = Compress.summarise_question(lb_prompts[i][0] + "\n" + lb_prompts[i][1])
        elif ranked_retrieve == True:
            prompt = Compress.ranked_retrieve(lb_prompts[i][0], lb_prompts[i][1]) + "\n" + lb_prompts[i][1]
        
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
        lb_output = response.choices[0].message.content
        
        elapsed = finish - start
        wall_times.append(elapsed)

        new_tokens = response.usage.completion_tokens
        tokens_per_second = new_tokens / (elapsed * pow(10, -9))
        token_rates.append(tokens_per_second)
        output_tokens.append(new_tokens)

        input_tokens.append(response.usage.prompt_tokens)

        # Reference for below code block: https://github.com/SafeAILab/EAGLE/issues/153
        #steps = int(output_ids[2])
        #avg_accept_len = new_tokens / steps
        #avg_accept_lens.append(avg_accept_len)

        # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
        output = {
            "id": hashlib.md5((prompt + lb_output).encode()).hexdigest(),
            "output": lb_output
        }
        LB_outputs.append(output)

# Print LongBench-E Results
if eagle3 == True:
    print(f"LongBench-E Results for {EAGLE_model_paths[0]}:")
else:
    print(f"LongBench-E Results for AutoReg {base_model_paths[0]}:")
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

if eagle3 == True:
    output_name = f"LBE_Output_EAGLE3_{EAGLE_model_paths[0]}{compression_tag}.jsonl" 
else:
    output_name = f"LBE_Output_AutoReg_{base_model_paths[0]}{compression_tag}.jsonl" 

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(output_name, "x") as f:
    for output in LB_outputs:
        f.write(json.dumps(output) + "\n")


# Final Plots

plt.title("Input Tokens vs Token Rates")
plt.plot(input_tokens, token_rates)

plt.title("Output Tokens vs Token Rates")
plt.plot(output_tokens, token_rates)


print("\n\n*******************************\nFinished Running Longbench_E_Test.py\n\n")

''' 
References
1.
'''