# KV Cache Experimentation
# Hyperparameters are "test_runs", "max_new_tokens", "temp"

import time
import numpy as np
import torch
import json
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import Data
import hashlib
import sys

def main(test_id):
    print("\n\n*******************************\nStarting KV_Cache_Test.py\n\n")

    base_model_paths = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
    EAGLE_model_paths = ["yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"]

    prompts = Data.specbench()

    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[0],
        ea_model_path=EAGLE_model_paths[0],
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    model.eval()

    KVTest_outputs = []
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

            elapsed = finish - start
            wall_times.append(elapsed)

            new_tokens = int(output_ids[1])
            tokens_per_second = new_tokens / (elapsed * pow(10, -9))
            token_rates.append(tokens_per_second)

            # Reference for below code block: https://github.com/SafeAILab/EAGLE/issues/153
            steps = int(output_ids[2])
            avg_accept_len = new_tokens / steps
            avg_accept_lens.append(avg_accept_len)

            # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
            output = {
                "id": hashlib.md5((str(test_run) + sb_prompt + Data.extract_response(kv_output)).encode()).hexdigest(),
                "output": Data.extract_response(kv_output)
            }
            KVTest_outputs.append(output)

    # Print KV Cache Test Results
    print(f"KV Cache Test Results for {EAGLE_model_paths[0]}:")
    print("Base Model: ", base_model_paths[0])
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))
    print("Average Acceptance Length: ", np.mean(avg_accept_lens))

    # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
    with open(f"KV_Cache_Test_Output_{EAGLE_model_paths[0].replace('/', '-')}_{test_id}.jsonl", "x") as f:
        for output in KVTest_outputs:
            f.write(json.dumps(output) + "\n")

    print("\n\nOutput Data: \n")

    for output in KVTest_outputs:
        print(output)

    print("\n\n*******************************\nFinished Running KV_Cache_Test.py\n\n")

if __name__ == "__main__":
    main(sys.argv[1])

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

5. C. W. F. Y. S. S. Y. W. Y. Z. Y. H. H. Z. Y. Z. Shenggui Li, Yikai Zhu, “Specforge: Train speculative decoding
models effortlessly,” https://github.com/sgl-project/specforge, 2025.

6. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang,
J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Proceedings of
the 37th International Conference on Neural Information Processing Systems, ser. NIPS ’23. Red Hook, NY,
USA: Curran Associates Inc., 2023. 

'''