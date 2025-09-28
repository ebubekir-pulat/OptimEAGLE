# KV Cache Experimentation
# Hyperparameters are "test_runs", "max_new_tokens", "temp"

print("\n\n*******************************\nStarting KV_Cache_Test.py\n\n")

import time
import numpy as np
import torch
import json
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import Data
import hashlib
import Compress

base_model_paths = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
EAGLE_model_paths = ["yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"]

def model_init():
    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[0],
        ea_model_path=EAGLE_model_paths[0],
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model


aai_ds = Data.aai_dataset()
AAI_outputs = []

# Hyperparameters
test_runs = 3
max_new_tokens = 512
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Translate: ", translate, "\n")

# AAI Dataset Assessment Loop
wall_times = []
token_rates = []
avg_accept_lens = []
model = model_init()

for test_run in range(test_runs):
    run = 1
    for question in aai_ds:
        print("Test Run: ", test_run)
        print("Test Question: ", run)
        run += 1

        # Below Code Block From: https://github.com/SafeAILab/EAGLE
        your_message = question
        if translate == True:
            your_message = Compress.zh_to_en(your_message)[0]
        conv = get_conversation_template(base_model_paths[0])
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()

        start = time.perf_counter_ns()

        # Below Code Line From: https://github.com/SafeAILab/EAGLE
        output_ids = model.eagenerate(input_ids, temperature=temp, max_new_tokens=max_new_tokens, log=True)

        finish = time.perf_counter_ns()

        if translate == True:
            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            aai_output = Compress.en_to_zh(model.tokenizer.decode(output_ids[0][0]))[0]
        else:
            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            aai_output = model.tokenizer.decode(output_ids[0][0])

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
            "id": hashlib.md5((str(test_run) + question + Data.extract_response(aai_output)).encode()).hexdigest(),
            "output": Data.extract_response(aai_output)
        }
        AAI_outputs.append(output)


# Print AAI Dataset Results
print(f"AAI Results for {EAGLE_model_paths[0]}:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens Generated/s: ", np.mean(token_rates))
print("Average Acceptance Length: ", np.mean(avg_accept_lens))

translate_tag = ""

if translate == True:
    translate_tag = "_Translate"

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(f"AAI_Output_EAGLE3{translate_tag}_{EAGLE_model_paths[0].replace("/", "-")}.jsonl", "x") as f:
    for output in AAI_outputs:
        f.write(json.dumps(output) + "\n")

print("\n\n*******************************\nFinished Running AAI_EAGLE3_Test.py\n\n")

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