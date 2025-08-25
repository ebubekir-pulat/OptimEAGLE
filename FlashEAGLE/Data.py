from datasets import load_dataset
import pandas as pd

def specbench():
    # Getting Spec-Bench Questions
    # Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
    jsonObj = pd.read_json(path_or_buf='../question.jsonl', lines=True)
    sb_prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]
    return sb_prompts


def aai_dataset():
    # Chinese (AAI) Dataset
    # Reference for below line: https://huggingface.co/datasets/PKU-Alignment/Align-Anything-Instruction-100K-zh
    aai_ds = load_dataset("PKU-Alignment/Align-Anything-Instruction-100K-zh", split="test")["prompt"]
    # Reference for above link

    return aai_ds


def longbench_e():
    # *** LongBench-E ***
    # Note - Reference LongBench-E in reference list
    # Getting LongBench-E Questions
    lb_prompts = []

    # Reference for below code block: https://www.w3schools.com/python/ref_list_sort.asp
    def sort_func(row):
        return len(row[0])

    # Reference for Below Code Block: https://huggingface.co/datasets/THUDM/LongBench 
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
                "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
        all_lb_prompts = []

        for i in range(len(data)):
            if data[i]["language"] != "zh":
                prompt = [data[i]["context"], data[i]["input"]]
                all_lb_prompts.append(prompt)
        
        # Reference for below code line: https://www.w3schools.com/python/ref_list_sort.asp
        all_lb_prompts.sort(key=sort_func)
        counter = 0

        for i in range(0, len(all_lb_prompts), 16):
            if counter == 15:
                break

            lb_prompts.append(all_lb_prompts[i])
            counter += 1
        
    return lb_prompts


def extract_response(model_output):
    response_index = model_output.find("### Assistant: ", 300) + len("### Assistant: ")
    model_output = model_output[response_index:]
    model_output = model_output[:model_output.find("### Human:")]
    model_output = model_output.strip()
    
    #print("\n\n*********************************************\nPrompt: ", question)
    #print("\nResponse: ", model_output)