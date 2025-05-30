import numpy as np
import pandas as pd    
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
jsonObj = pd.read_json(path_or_buf='../../Data/question.jsonl', lines=True)
prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]

LLM_pairs = [["lmsys/vicuna-13b-v1.3", "double7/vicuna-68m"],  # [target model, draft model]
             ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "JackFram/llama-68m"]]

# Below Code Block From: https://huggingface.co/blog/assisted-generation
checkpoint = LLM_pairs[1][0]
assistant_checkpoint = LLM_pairs[1][1]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)

template = ""
if "vicuna" in checkpoint:
    template = "vicuna"
else:
    template = "llama-3-chat"

wall_times = []
token_rates = []

# Below Code Line From: https://github.com/SafeAILab/EAGLE
model.eval()

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

        # Below Code Line From: https://huggingface.co/blog/assisted-generation
        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.perf_counter_ns()

        # 2 Code Lines Below From: https://huggingface.co/blog/assisted-generation
        outputs = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=512)
        #print("Output: ", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        finish = time.perf_counter_ns()
        elapsed = finish - start
        wall_times.append(elapsed)
        #print("Wall Clock Time (ns): ", elapsed)

        num_tokens = len(outputs)
        tokens_per_second = num_tokens / (elapsed * pow(10, -9))
        token_rates.append(tokens_per_second)
        #print("Tokens Per Second: ", tokens_per_second)

print("Results:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens/s: ", np.mean(token_rates))

'''
References

1. Gante, J 2023, 'Assisted Generation: a new direction toward low-latency text generation', Hugging Face Blog, 11 May, <https://huggingface.co/blog/assisted-generation>.

2. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, 'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty', in International Conference on Machine Learning.

3. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, 'EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees', in Empirical Methods in Natural Language Processing.

4. Li, Y, Wei, F, Zhang, C & Zhang, H 2025, 'EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test', <https://arxiv.org/abs/2503.01840>.

@misc {gante2023assisted,
    author       = { {Joao Gante} },
    title        = { Assisted Generation: a new direction toward low-latency text generation },
    year         = 2023,
    url          = { https://huggingface.co/blog/assisted-generation },
    doi          = { 10.57967/hf/0638 },
    publisher    = { Hugging Face Blog }
}
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