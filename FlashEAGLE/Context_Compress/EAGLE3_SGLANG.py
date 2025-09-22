# Standard EAGLE-3 SGLANG
# Hyperparameters: dataset, test_runs, max_new_tokens, temp

print("\n\n*******************************\nStarting EAGLE3_SGLANG.py\n\n")

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import hashlib

base_model_paths = ["Qwen/Qwen3-8B"]
EAGLE_model_paths = ["Tengyunw/qwen3_8b_eagle3"]

# Hyperparameter
dataset = "THUDM/LongBench"
if dataset == "THUDM/LongBench":
    prompts = Data.longbench_e()
elif dataset == "PKU-Alignment/Align-Anything-Instruction-100K-zh":
    prompts = Data.aai_dataset()
elif dataset == "SpecBench":
    prompts = Data.specbench()

# Preparing SGLANG with EAGLE3
# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
server_process, port = launch_server_cmd(
    f"""
python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path {EAGLE_model_paths[0]} --dtype float16
"""
)

# Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
wait_for_server(f"http://localhost:{port}")
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

eagle3_outputs = []
# Hyperparameters
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Dataset: ", dataset)
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)

# EAGLE3 Assessment Loop
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

        if dataset == "THUDM/LongBench":
            prompt = prompts[i][0] + "\n" + prompts[i][1]
        elif dataset == "PKU-Alignment/Align-Anything-Instruction-100K-zh":
            prompt = prompts[i]
        elif dataset == "SpecBench":
            prompt = prompts[i][0]
        
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
        eagle3_output = response.choices[0].message.content
        
        elapsed = finish - start
        wall_times.append(elapsed)

        new_tokens = response.usage.completion_tokens
        tokens_per_second = new_tokens / (elapsed * pow(10, -9))
        token_rates.append(tokens_per_second)
        output_tokens.append(new_tokens)

        input_tokens.append(response.usage.prompt_tokens)

        # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
        output = {
            "id": hashlib.md5((str(test_run) + prompt + eagle3_output).encode()).hexdigest(),
            "output": eagle3_output
        }
        eagle3_outputs.append(output)

# Print EAGLE-3 Results
print(f"EAGLE-3 for {EAGLE_model_paths[0]}:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens Generated/s: ", np.mean(token_rates))

# Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
terminate_process(server_process)

output_name = f"EAGLE3_Output_{EAGLE_model_paths[0].replace("/", "-")}_{dataset}.jsonl" 

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(output_name, "x") as f:
    for output in eagle3_outputs:
        f.write(json.dumps(output) + "\n")


print("\n\n*******************************\nFinished Running EAGLE3_SGLANG.py\n\n")

''' 
References

1. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE: Speculative sampling requires rethinking feature
uncertainty,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings
of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett,
and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 28 935–28 948. [Online]. Available:
https://proceedings.mlr.press/v235/li24bt.html

2. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE-2: Faster inference of language models with dynamic
draft trees,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 7421–7432. [Online]. Available: https://aclanthology.org/2024.emnlp-main.422/

3. Y. Li, F. Wei, C. Zhang, and H. Zhang, “Eagle-3: Scaling up inference acceleration of large language models
via training-time test,” 2025. [Online]. Available: https://arxiv.org/abs/2503.01840

4. C. W. F. Y. S. S. Y. W. Y. Z. Y. H. H. Z. Y. Z. Shenggui Li, Yikai Zhu, “Specforge: Train speculative decoding
models effortlessly,” https://github.com/sgl-project/specforge, 2025.

5. Q. Team, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

'''