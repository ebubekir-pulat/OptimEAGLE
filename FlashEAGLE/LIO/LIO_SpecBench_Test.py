# LIO Testing on Spec-Bench
# Hyperparameters: summarise, test_runs, max_new_tokens, temp

print("\n\n*******************************\nStarting LIO_SpecBench_Test.py\n\n")

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import Compress
import hashlib
import matplotlib.pyplot as plt

LIO_model_paths = ["openai/gpt-oss-20b"]
base_model_paths = ["Qwen/Qwen3-8B"]
EAGLE_model_paths = ["Tengyunw/qwen3_8b_eagle3"]

prompts = Data.specbench()
tasks = Data.specbench_tasks()
tasks_set = {tasks}
optim_params = {}

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

for task in tasks_set:

    LIO_prompt = f"Generate optimal hyperparameters for EAGLE-3 speculative decoding with SGLANG, where the \
                base model to be used is {base_model_paths[0]}, the EAGLE-3 model to be used is {EAGLE_model_paths[0]}, \
                the dataset to be tested on is Spec-Bench and the specific task type is {task}. Choose hyperparameters that \
                optimise acceptance length, tokens generated per second and wall-time speedup. Provide as many hyperparameters \
                as necessary for maximum performance. Generate the hyperparameters in the format: \
                --hyperparameter_name1 value --hyperparameter_name2 value and so on. Before providing the hyperparameters, \
                put a #START delimiter, and when finished, put a #END delimiter."

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
    LIO_output = Data.extract_LIO_response(LIO_output)
    optim_params[task] = LIO_output

# Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
terminate_process(server_process)

servers = {}

for task in optim_params:
    # Preparing SGLANG with EAGLE3
    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    server_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path {EAGLE_model_paths[0]} {optim_params[task]} --dtype float16
    """
    )

    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    servers[task] = [server_process, port, client]


LIO_outputs = []
# Hyperparameters
summarise = False
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Dataset: Spec-Bench")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Summarise: ", summarise, "\n")

# LIO Assessment Loop
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

        prompt = prompts[i][0]

        if summarise == True:
            prompt = Compress.summarise_text(prompts[i][0])
        
        server_process, port, client = servers[tasks[i]]
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

        input_tokens.append(response.usage.prompt_tokens)

        # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
        output = {
            "id": hashlib.md5((str(test_run) + prompt + model_output).encode()).hexdigest(),
            "output": model_output
        }
        LIO_outputs.append(output)

# Print LIO Results
print(f"LIO Spec-Bench Results for {LIO_model_paths[0]}:")
print(f"Dataset: Spec-Bench")
print(f"EAGLE Model: {EAGLE_model_paths[0]}")
print(f"Base Model: {base_model_paths[0]}")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens Generated/s: ", np.mean(token_rates))

for task in servers:
    server_process, port, client = servers[task]

    # Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    terminate_process(server_process)

compression_tag = ""
if summarise == True:
    compression_tag = "_Summ"

output_name = f"LIO_SB_Output_{LIO_model_paths[0].replace("/", "-")}_{EAGLE_model_paths[0].replace("/", "-")}{compression_tag}.jsonl" 
 
# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(output_name, "x") as f:
    for output in LIO_outputs:
        f.write(json.dumps(output) + "\n")

# Final Plots
plt.title("Input Tokens vs Token Rates")
plt.plot(input_tokens, token_rates)
plt.savefig("InputTokens_vs_TokenRates.png")

plt.title("Output Tokens vs Token Rates")
plt.plot(output_tokens, token_rates)
plt.savefig("OutputTokens_vs_TokenRates.png")


print("\n\n*******************************\nFinished Running LIO_SpecBench_Test.py\n\n")

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

5. OpenAI, “gpt-oss-120b gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

6. Q. Team, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

'''