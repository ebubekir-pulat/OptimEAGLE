# Code For KV Cache Experimentation
# Hyperparameters are "test_runs", "max_new_tokens", "temp", "test_id"
# test_id defines the KV Cache Test ID

import time
import numpy as np
import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import Data
import sys
import subprocess

subprocess.run(
    ["nvidia-smi"], check=True
)

def main(test_id):
    print("\n\n*******************************\nStarting KV_Cache_Test.py\n\n")

    base_model_paths = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    EAGLE_model_paths = ["yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"]

    prompts = Data.specbench()
    print("Spec-Bench Shape: ", np.shape(prompts))

    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[0],
        ea_model_path=EAGLE_model_paths[0],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    KVTest_outputs = []
    KVTest_parsed_outputs = []
    # Hyperparameters
    test_runs = 3
    max_new_tokens = 1024
    temp = 0.0

    print("\nEvaluation Settings Chosen:")
    print("Test Runs: ", test_runs)
    print("Max New Tokens: ", max_new_tokens)
    print("Temperature: ", temp, "\n")

    # KV Cache Test Assessment Loop
    wall_times = []
    token_rates = []
    avg_accept_lens = []

    for test_run in range(test_runs):
        run = 1
        for i in range(len(prompts)):
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            sb_prompt = prompts[i][0]

            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            conv = get_conversation_template(base_model_paths[0])
            conv.append_message(conv.roles[0], sb_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()

            start = time.perf_counter_ns()

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            output_ids = model.eagenerate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, log=True)

            finish = time.perf_counter_ns()

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            kv_output = model.tokenizer.decode(output_ids[0][0])

            #print("Original Output: ", kv_output)
            #print("\nParsed Output: ", Data.extract_response(kv_output), end="\n")

            elapsed = finish - start
            wall_times.append(elapsed)

            new_tokens = int(output_ids[1])
            tokens_per_second = new_tokens / (elapsed * pow(10, -9))
            token_rates.append(tokens_per_second)

            # Reference for below code block: https://github.com/SafeAILab/EAGLE/issues/153
            steps = output_ids[2]
            avg_accept_len = new_tokens / steps
            avg_accept_lens.append(avg_accept_len)

            KVTest_parsed_outputs.append(Data.extract_response(kv_output))
            KVTest_outputs.append(kv_output)

    # Print KV Cache Test Results
    print(f"KV Cache Test Results for {EAGLE_model_paths[0]}:")
    print("Test ID: ", test_id)
    print("Base Model: ", base_model_paths[0])
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))
    print("Average Acceptance Length: ", np.mean(avg_accept_lens))

    print("\n\nRaw Output Data: \n")
    print(KVTest_outputs)

    print("\n\nParsed Output Data: \n")
    print(KVTest_parsed_outputs)

    print("\n\n*******************************\nFinished Running KV_Cache_Test.py\n\n")

if __name__ == "__main__":
    main(int(sys.argv[1]))

'''
References

1. DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025.
[Online]. Available: https://arxiv.org/abs/2501.12948

2. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE: Speculative sampling requires rethinking feature
uncertainty,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings
of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett,
and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 28 935–28 948. [Online]. Available:
https://proceedings.mlr.press/v235/li24bt.html

3. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE-2: Faster inference of language models with dynamic
draft trees,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 7421–7432. [Online]. Available: https://aclanthology.org/2024.emnlp-main.422/

4. Y. Li, F. Wei, C. Zhang, and H. Zhang, “Eagle-3: Scaling up inference acceleration of large language models
via training-time test,” 2025. [Online]. Available: https://arxiv.org/abs/2503.01840

5. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang,
J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Proceedings of
the 37th International Conference on Neural Information Processing Systems, ser. NIPS ’23. Red Hook, NY,
USA: Curran Associates Inc., 2023. 

'''