import pandas as pd
import numpy as np
from transformers import AutoTokenizer,  AutoModelForCausalLM
import time
from fastchat.model import get_conversation_template    

# Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
jsonObj = pd.read_json(path_or_buf='../../Data/question.jsonl', lines=True)
prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]

LLMs = ["lmsys/vicuna-7b-v1.3",
        "lmsys/vicuna-13b-v1.3",
        "lmsys/vicuna-33b-v1.3",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "meta-llama/Llama-3.3-70B-Instruct"]

# Below Code Block From: https://huggingface.co/learn/llm-course/chapter2/6?fw=pt, https://huggingface.co/docs/hub/transformers
checkpoint = LLMs[3]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# Below Code Line From: https://github.com/SafeAILab/EAGLE
model.eval()

template = ""
if "vicuna" in checkpoint:
    template = "vicuna"
else:
    template = "llama-3-chat"

wall_times = []
token_rates = []

for _ in range(3):
    for i in range(160, 480):
        # Below Code Block From: https://github.com/SafeAILab/EAGLE
        # Code Block Starts Here
        your_message = prompts[i]
        if len(your_message) == 1: 
            your_message = your_message[0]
        else: 
            raise("Message Length Above 1")
        conv = get_conversation_template(template)
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # Code Blocks Ends Here

        # Below Code Block From: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM, https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM.forward.example
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.perf_counter_ns()
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=512)
        #print("\n\nOUTPUT: ", tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

        finish = time.perf_counter_ns()
        elapsed = finish - start
        wall_times.append(elapsed)
        #print("\nWall Clock Time (ns): ", elapsed)

        num_tokens = len(generate_ids)
        tokens_per_second = num_tokens / (elapsed * pow(10, -9))
        token_rates.append(tokens_per_second)
        #print("Tokens Per Second: ", tokens_per_second)

print("Results:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens/s: ", np.mean(token_rates))

'''
References

1. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, '{EAGLE}: Speculative Sampling Requires Rethinking Feature Uncertainty', International Conference on Machine Learning.

2. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, '{EAGLE-2}: Faster Inference of Language Models with Dynamic Draft Trees', Empirical Methods in Natural Language Processing.

3. Li, Y, Wei, F, Zhang, C & Zhang, H 2025, '{EAGLE-3}: Scaling up Inference Acceleration of Large Language Models via Training-Time Test', <https://arxiv.org/abs/2503.01840>.

@inproceedings{li2024eagle, 
	author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
	title = {{EAGLE}: Speculative Sampling Requires Rethinking Feature Uncertainty}, 
	booktitle = {International Conference on Machine Learning},
	year = {2024}
}
@inproceedings{li2024eagle2, 
	author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
	title = {{EAGLE-2}: Faster Inference of Language Models with Dynamic Draft Trees}, 
	booktitle = {Empirical Methods in Natural Language Processing},
	year = {2024}
}
@misc{li2025eagle3scalinginferenceacceleration,
      title={{EAGLE-3}: Scaling up Inference Acceleration of Large Language Models via Training-Time Test}, 
      author={Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang},
      year={2025},
      eprint={2503.01840},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01840}, 
}
'''