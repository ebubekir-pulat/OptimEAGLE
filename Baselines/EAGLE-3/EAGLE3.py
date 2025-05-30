import pandas as pd    
import time
import numpy as np
import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template

# Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
jsonObj = pd.read_json(path_or_buf='../../Data/question.jsonl', lines=True)
prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]

base_model_paths = ["lmsys/vicuna-7b-v1.3",
                    "lmsys/vicuna-13b-v1.3",
                    "lmsys/vicuna-33b-v1.3",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "meta-llama/Llama-3.3-70B-Instruct"]

EAGLE_model_paths = ["yuhuili/EAGLE-Vicuna-7B-v1.3",
                     "yuhuili/EAGLE3-Vicuna1.3-13B",
                     "yuhuili/EAGLE-Vicuna-33B-v1.3",
                     "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
                     "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"]

model_index = 3
template = ""
if "vicuna" in base_model_paths[model_index]:
    template = "vicuna"
else:
    template = "llama-3-chat"

wall_times = []
token_rates = []

# Below Code Block From https://github.com/SafeAILab/EAGLE
model = EaModel.from_pretrained(
    base_model_path=base_model_paths[model_index],
    ea_model_path=EAGLE_model_paths[model_index],
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    total_token=-1
    # offload_folder="offload" # Code Line From: https://github.com/nomic-ai/gpt4all/issues/239
)

# Below Code Line From https://github.com/SafeAILab/EAGLE
model.eval()

for _ in range(3):
    for i in range(160, 480):
        # Below Code Block From https://github.com/SafeAILab/EAGLE
        your_message = prompts[i]
        if len(your_message) == 1: 
            your_message = your_message[0]
        else: 
            raise("Message Length Above 1")
        conv = get_conversation_template(template)
        conv.append_message(conv.roles[0], your_message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids=model.tokenizer([prompt]).input_ids
        input_ids = torch.as_tensor(input_ids).cuda()

        start = time.perf_counter_ns()

        # Below Code Line From https://github.com/SafeAILab/EAGLE
        output_ids=model.eagenerate(input_ids, temperature=0.0, max_new_tokens=512)
        #output=model.tokenizer.decode(output_ids[0])

        finish = time.perf_counter_ns()
        elapsed = finish - start
        wall_times.append(elapsed)
        #print("Wall Clock Time (ns): ", elapsed)

        num_tokens = len(output_ids)
        tokens_per_second = num_tokens / (elapsed * pow(10, -9))
        token_rates.append(tokens_per_second)
        #print("Tokens Per Second: ", tokens_per_second)

print("Results:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens/s: ", np.mean(token_rates))

'''
References

1. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, 'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty', in International Conference on Machine Learning.

2. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, 'EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees', in Empirical Methods in Natural Language Processing.

3. Li, Y, Wei, F, Zhang, C & Zhang, H 2025, 'EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test', <https://arxiv.org/abs/2503.01840>.

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