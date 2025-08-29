print("\n\n*******************************\nStarting LBE_Solo_Test.py\n\n")

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import Compress
import hashlib

EAGLE_model_paths = ["AngelSlim/Qwen3-1.7B_eagle3"]
# Note: Reference for Qwen3: https://huggingface.co/Qwen/Qwen3-1.7B, https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3

models_to_test = [0]
lb_prompts = Data.longbench_e()

# Preparing SGLANG with EAGLE3
# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
server_process, port = launch_server_cmd(
    f"""
python3 -m sglang.launch_server --model {EAGLE_model_paths[0]} \
        --mem-fraction 0.6 \
        --cuda-graph-max-bs 2 --dtype float16
"""
)
wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")


LB_outputs = []
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
for model_index in models_to_test:
    wall_times = []
    #token_rates = []
    #avg_accept_lens = []
    
    for test_run in range(test_runs):
        run = 1
        for i in range(len(lb_prompts)):
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            start = time.perf_counter_ns()

            prompt = lb_prompts[i][0] + "\n" + lb_prompts[i][1]
            if summarise == True:
                prompt = Compress.summarise_question(lb_prompts[i][0] + "\n" + lb_prompts[i][1])
            elif ranked_retrieve == True:
                prompt = Compress.ranked_retrieve(lb_prompts[i][0], lb_prompts[i][1]) + "\n" + lb_prompts[i][1]
            
            # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
            response = client.chat.completions.create(
                model=EAGLE_model_paths[0],
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_new_tokens,
            )

            # Reference for below code line: https://stackoverflow.com/questions/77444332/openai-python-package-error-chatcompletion-object-is-not-subscriptable 
            lb_output = response.choices[0].message.content
            
            finish = time.perf_counter_ns()
            elapsed = finish - start
            wall_times.append(elapsed)

            #new_tokens = int(output_ids[1])
            #tokens_per_second = new_tokens / (elapsed * pow(10, -9))
            #token_rates.append(tokens_per_second)

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
    print(f"LongBench-E Results for {EAGLE_model_paths[0]}:")
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    #print("Mean Tokens Generated/s: ", np.mean(token_rates))
    #print("Average Acceptance Length: ", np.mean(avg_accept_lens))

# Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
terminate_process(server_process)

compression_tag = ""

if summarise == True:
    compression_tag = "_Summ"
elif ranked_retrieve == True:
    compression_tag == "_RR"

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(f"LBE_Solo_Output_{EAGLE_model_paths[0]}{compression_tag}.jsonl", "w") as f:
    for output in LB_outputs:
        f.write(json.dumps(output) + "\n")


print("\n\n*******************************\nFinished Running LBE_Solo_Test.py\n\n")

''' 
References
1.
'''