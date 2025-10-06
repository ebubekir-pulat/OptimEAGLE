from datasets import load_dataset
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


def aai_dataset():
    # Chinese (AAI) Dataset
    # Reference for below line: https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K-zh
    aai_ds = load_dataset("PKU-Alignment/Align-Anything-Instruction-100K-zh", split="test")["prompt"]
    return aai_ds


def longbench_e():
    # Getting LongBench-E Questions
    lb_prompts = []

    # Reference for below code block: https://www.w3schools.com/python/ref_list_sort.asp
    def sort_func(row):
        return len(row[0])

    # Reference for Below Code Block: https://huggingface.co/datasets/THUDM/LongBench 
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
                "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', trust_remote_code=True)
        all_lb_prompts = []

        for i in range(len(data)):
            if data[i]["language"] == "en":
                prompt = [data[i]["context"], data[i]["input"]]
                all_lb_prompts.append(prompt)
        
        # Reference for below code line: https://www.w3schools.com/python/ref_list_sort.asp
        all_lb_prompts.sort(key=sort_func)
        counter = 0

        for i in range(0, len(all_lb_prompts), 11):
            if counter == 19:
                break

            lb_prompts.append(all_lb_prompts[i])
            counter += 1
        
    return lb_prompts


def extract_response(model_output):
    response_index = model_output.find("### Assistant: ", 300) + len("### Assistant: ")
    model_output = model_output[response_index:]
    model_output = model_output[:model_output.find("### Human:")]
    model_output = model_output.strip()
    return model_output

def extract_LIO_response_old(model_output):
    # Reference for below code block: https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    num_steps = model_output[model_output.find('speculative-num-steps: ') + len('speculative-num-steps: ') : model_output.find('#1')]
    topk = model_output[model_output.find('speculative-eagle-topk: ') + len('speculative-eagle-topk: ') : model_output.find('#2')]
    draft_tokens = model_output[model_output.find('speculative-num-draft-tokens: ') + len('speculative-num-draft-tokens: ') : model_output.find('#3')]
    graph_max_bs = model_output[model_output.find('cuda-graph-max-bs: ') + len('cuda-graph-max-bs: ') : model_output.find('#4')]

    return num_steps, topk, draft_tokens, graph_max_bs

def extract_LIO_response(model_output):
    # Reference for below code line: https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    model_output = model_output[model_output.rfind('#START') + len('#START') : model_output.rfind('#END')]
    return model_output


'''
References

1. Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and
J. Li, “LongBench: A bilingual, multitask benchmark for long context understanding,” in Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku,
A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug.
2024, pp. 3119–3137. [Online]. Available: https://aclanthology.org/2024.acl-long.172/

2. Q. Lhoest, A. Villanova del Moral, Y. Jernite, A. Thakur, P. von Platen, S. Patil, J. Chaumond, M. Drame,
J. Plu, L. Tunstall, J. Davison, M. ˇSaˇsko, G. Chhablani, B. Malik, S. Brandeis, T. Le Scao, V. Sanh, C. Xu,
N. Patry, A. McMillan-Major, P. Schmid, S. Gugger, C. Delangue, T. Matussi`ere, L. Debut, S. Bekman, P. Cistac,
T. Goehringer, V. Mustar, F. Lagunas, A. Rush, and T. Wolf, “Datasets: A community library for natural language
processing,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing:
System Demonstrations. Online and Punta Cana, Dominican Republic: Association for Computational
Linguistics, Nov. 2021, pp. 175–184. [Online]. Available: https://aclanthology.org/2021.emnlp-demo.21

3. H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui, “Unlocking efficiency in large
language model inference: A comprehensive survey of speculative decoding,” in Findings of the Association
for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok,
Thailand: Association for Computational Linguistics, Aug. 2024, pp. 7655–7671. [Online]. Available:
https://aclanthology.org/2024.findings-acl.456/

'''