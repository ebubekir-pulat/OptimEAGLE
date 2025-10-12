# LIO Dataset-Specific Testing (with Speculative Decoding Parameters Focus)
# Hyperparameters: test_runs, max_new_tokens, temp

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

def main():
    print("\n\n*******************************\nStarting LIO_DatasetSpecific_Spec_Test.py\n\n")

    print("Python Version:")

    subprocess.run(
        ["python", "--version"], check=True
    )

    LIO_model_paths = ["openai/gpt-oss-20b"]
    base_model_paths = ["Qwen/Qwen3-8B"]
    EAGLE_model_paths = ["Tengyunw/qwen3_8b_eagle3"]
    
    prompts = Data.specbench()
    print("Spec-Bench Dataset Shape: ", np.shape(prompts))

    LIO_prompt = f"Generate optimal hyperparameters for EAGLE-3 speculative decoding with SGLANG, where the \
                base model to be used is {base_model_paths[0]}, the EAGLE-3 model to be used is {EAGLE_model_paths[0]} \
                and the dataset to be tested on is Spec-Bench. Spec-Bench is a benchmark covering multi-turn conversation, \
                translation, summarisation, question answering, mathematical reasoning and retrieval-augmented generation, \
                consisting of samples from the MT-bench, WMT14 DE-EN, CNN/Daily Mail, Natural Questions, GSM8K and DPR \
                datasets. Provide hyperparameters that optimise acceptance length, tokens generated per second and wall-time speedup. \
                Specifically, provide values for these parameters: --speculative-num-steps, \
                --speculative-eagle-topk, --speculative-num-draft-tokens and --speculative-attention-mode (prefill or decode). \
                Avoid extreme values that can cause errors. Generate the hyperparameters in the format --parameter-name value, with spaces in between. \
                Before providing the hyperparameters, put a #START delimiter, and when finished, put a #END delimiter. THIS IS IMPORTANT. \
                Make sure to follow the format, and ensure your total output is within 8192 tokens MAXIMUM!"

    print("LIO Prompt:\n", LIO_prompt, "\nEND OF LIO PROMPT")

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
        max_tokens=8192,
    )

    # Reference for below code line: https://stackoverflow.com/questions/77444332/openai-python-package-error-chatcompletion-object-is-not-subscriptable 
    LIO_output = response.choices[0].message.content

    # Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    terminate_process(server_process)

    print("LIO Output Before Processing:\n", LIO_output, "\nEND OF LIO OUTPUT")
    LIO_output = Data.extract_LIO_response(LIO_output)
    print("\n\nLIO Output:\n", LIO_output, "\nEND OF LIO OUTPUT")

    # Preparing SGLANG with EAGLE3
    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    server_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path {EAGLE_model_paths[0]} {LIO_output}
    """
    )

    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    LIO_outputs = []
    # Hyperparameters
    test_runs = 3
    max_new_tokens = 1024
    temp = 0.0

    print("\nEvaluation Settings Chosen:")
    print("Dataset: Spec-Bench")
    print("Test Runs: ", test_runs)
    print("Max New Tokens: ", max_new_tokens)
    print("Temperature: ", temp, "\n")

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
            LIO_outputs.append(model_output)

    # Print LIO Results
    print(f"LIO Results for {LIO_model_paths[0]}:")
    print(f"Dataset: Spec-Bench")
    print(f"EAGLE Model: {EAGLE_model_paths[0]}")
    print(f"Base Model: {base_model_paths[0]}")
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))

    # Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    terminate_process(server_process)

    print("Walltimes Array: ", wall_times)
    print("Input Tokens Array: ", input_tokens)
    print("Output Tokens Array: ", output_tokens)
    print("Tokens Generated Per Second Array: ", token_rates)

    print("\n\nOutput Data: \n")
    print(LIO_outputs)

    print("\n\n*******************************\nFinished Running LIO_DatasetSpecific_Spec_Test.py\n\n")

if __name__ == "__main__":
    main()

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

4. OpenAI, “gpt-oss-120b gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

5. Q. Team, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

'''