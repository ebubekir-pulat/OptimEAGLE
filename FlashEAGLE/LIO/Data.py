import pandas as pd

def specbench():
    # Get Spec-Bench Data from: https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl 
    # Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
    jsonObj = pd.read_json(path_or_buf='question.jsonl', lines=True)
    sb_prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]
    sb_prompts = sb_prompts[80:]
    return sb_prompts

def specbench_tasks():
    # Get Spec-Bench Data from: https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl 
    # Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
    jsonObj = pd.read_json(path_or_buf='question.jsonl', lines=True)
    sb_tasks = [jsonObj.at[i, 'category'] for i in range(len(jsonObj))]
    sb_tasks = sb_tasks[80:]
    return sb_tasks

'''

def extract_response(model_output):
    response_index = model_output.find("### Assistant: ", 300) + len("### Assistant: ")
    model_output = model_output[response_index:]
    model_output = model_output[:model_output.find("### Human:")]
    model_output = model_output.strip()
    return model_output

'''

def extract_LIO_response(model_output):
    # Reference for below code line: https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    model_output = model_output[model_output.rfind('#START') + len('#START') : model_output.rfind('#END')]
    return model_output


'''
References

1. H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui, “Unlocking efficiency in large
language model inference: A comprehensive survey of speculative decoding,” in Findings of the Association
for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok,
Thailand: Association for Computational Linguistics, Aug. 2024, pp. 7655–7671. [Online]. Available:
https://aclanthology.org/2024.findings-acl.456/

'''