from vllm import LLM, SamplingParams
import pandas as pd
import time

# Getting Spec-Bench Questions
# Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
jsonObj = pd.read_json(path_or_buf='../question.jsonl', lines=True)
sb_prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]

# Code in Rest of File From: https://docs.vllm.ai/en/stable/features/spec_decode.html#speculating-using-eagle-based-draft-models
sampling_params = SamplingParams(temperature=0.0, max_new_tokens=256)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 5,
    },
)

start = time.perf_counter_ns()

outputs = llm.generate(sb_prompts, sampling_params)

finish = time.perf_counter_ns()
elapsed = finish - start
print("Wall Time (ns): ", elapsed)

#for output in outputs:
#    prompt = output.prompt
#    generated_text = output.outputs[0].text
#    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")