# Context Compression Testing on Align-Anything-Instruction-100K-zh

print("\n\n*******************************\nStarting AAI_AutoReg_Test.py\n\n")

import time
import numpy as np
import json
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import Data
import hashlib
import Compress

base_model_paths = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]

def model_init():
    # Below Code Block From: https://huggingface.co/learn/llm-course/chapter2/6?fw=pt, https://huggingface.co/docs/hub/transformers
    checkpoint = base_model_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model, tokenizer

aai_ds = Data.aai_dataset()
AAI_outputs = []
translate = True
test_runs = 1
max_new_tokens = 128
temp = 0.0

print("\nEvaluation Settings Chosen:")
print("Test Runs: ", test_runs)
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp)
print("Translate: ", translate, "\n")

# AAI Dataset Assessment Loop
wall_times = []
model, tokenizer = model_init()

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
        
        # Below Code Line From: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM, https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM.forward.example
        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.perf_counter_ns()
        
        # Below Code Line From: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM, https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM.forward.example
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens) 

        finish = time.perf_counter_ns()

        # Reference for below code line: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM.forward.example
        decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if translate == True:
            aai_output = Compress.en_to_zh(decoded_output)[0]
        else:
            aai_output = decoded_output

        elapsed = finish - start
        wall_times.append(elapsed)

        # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
        output = {
            "id": hashlib.md5((question + Data.extract_response(aai_output)).encode()).hexdigest(),
            "output": Data.extract_response(aai_output)
        }
        AAI_outputs.append(output)


# Print AAI Dataset Results
print(f"AAI Results for {base_model_paths[0]}:")
print("Mean Wall Time (ns): ", np.mean(wall_times))

translate_tag = ""

if translate == True:
    translate_tag = "_Translate"

# Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
with open(f"AAI_Output_AutoReg{translate_tag}_{base_model_paths[0].replace("/", "-")}.jsonl", "x") as f:
    for output in AAI_outputs:
        f.write(json.dumps(output) + "\n")

print("\n\n*******************************\nFinished Running AAI_AutoReg_Test.py\n\n")


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

5. T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz,
J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger,
M. Drame, Q. Lhoest, and A. M. Rush, “Transformers: State-of-the-art natural language processing,”
in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations. Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45. [Online]. Available:
https://www.aclweb.org/anthology/2020.emnlp-demos.6

6. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang,
J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Proceedings of
the 37th International Conference on Neural Information Processing Systems, ser. NIPS ’23. Red Hook, NY,
USA: Curran Associates Inc., 2023. 

'''