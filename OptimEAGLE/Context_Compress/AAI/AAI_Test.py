# Context Compression Testing on Align-Anything-Instruction-100K-zh
# Translation
# Hyperparameters: eagle3, translate, test_runs, max_new_tokens, temp

import subprocess

subprocess.run(
    ["sudo", "apt-get", "-y", "install", "libnuma-dev"], check=True
)

subprocess.run(
    ["nvidia-smi"], check=True
)

import time
import numpy as np
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import Compress
import sys

def main(eagle3, translate):
    print("\n\n*******************************\nStarting AAI_Test.py\n\n")

    if eagle3 == "True":
        eagle3 = True
    else:
        eagle3 = False

    if translate == "True":
        translate = True
    else:
        translate = False

    print("Python Version:")

    subprocess.run(
        ["python", "--version"], check=True
    )

    base_model_paths = ["Qwen/Qwen3-8B"]
    EAGLE_model_paths = ["Tengyunw/qwen3_8b_eagle3"]

    aai_ds = Data.aai_dataset()
    print("AAI Dataset Shape: ", np.shape(aai_ds))

    if eagle3 == True:
        # Preparing SGLANG with EAGLE3
        # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
        server_process, port = launch_server_cmd(
            f"""
        python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path {EAGLE_model_paths[0]} --dtype float16
        """
        )
    else:
        # Preparing AutoReg SGLANG
        # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html, https://docs.sglang.ai/basic_usage/send_request.html
        server_process, port = launch_server_cmd(
            f"""
        python3 -m sglang.launch_server --model-path {base_model_paths[0]} --dtype float16
        """
        )

    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    AAI_outputs = []
    # Hyperparameters
    test_runs = 3
    max_new_tokens = 512
    temp = 0.0

    print("\nEvaluation Settings Chosen:")
    print("EAGLE3: ", eagle3)
    print("Translate: ", translate)
    print("Test Runs: ", test_runs)
    print("Max New Tokens: ", max_new_tokens)
    print("Temperature: ", temp, "\n")

    # AAI Assessment Loop
    wall_times = []
    token_rates = []
    input_tokens = []
    output_tokens = []

    for test_run in range(test_runs):
        run = 1
        for prompt in aai_ds:
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            if translate == True:
                prompt = Compress.zh_to_en(prompt)[0]
            
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
            aai_output = response.choices[0].message.content

            if translate == True:
                aai_output = Compress.en_to_zh(aai_output)[0]
            
            elapsed = finish - start
            wall_times.append(elapsed)

            new_tokens = response.usage.completion_tokens
            tokens_per_second = new_tokens / (elapsed * pow(10, -9))
            token_rates.append(tokens_per_second)
            output_tokens.append(new_tokens)

            input_tokens.append(response.usage.prompt_tokens)
            AAI_outputs.append(aai_output)

    # Print AAI Results
    if eagle3 == True:
        print(f"AAI Results for {EAGLE_model_paths[0]}:")
    else:
        print(f"AAI Results for {base_model_paths[0]}:")
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))

    # Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    terminate_process(server_process)

    print("Input Tokens Array: ", input_tokens)
    print("Output Tokens Array: ", output_tokens)
    print("Tokens Generated Per Second Array: ", token_rates)
    print("Walltimes Array: ", wall_times)

    print("\n\nOutput Data: \n")
    print(AAI_outputs)

    print("\n\n*******************************\nFinished Running AAI_Test.py\n\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


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

4. Q. Team, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

'''