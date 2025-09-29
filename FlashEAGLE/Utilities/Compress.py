from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import nltk
import torch
import numpy as np

# Preparing Translater
# Reference for below code block: https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_small_1024 
model_name = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
translate_model = T5ForConditionalGeneration.from_pretrained(model_name)
translate_model.to('cuda')
translate_tokenizer = T5Tokenizer.from_pretrained(model_name)

def zh_to_en(text):
    # Reference for below code block: https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_small_1024
    input_ids = translate_tokenizer("translate to en: " + text, return_tensors="pt")
    generated_tokens = translate_model.generate(**input_ids.to('cuda'))
    return translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) 

def en_to_zh(text):
    # Reference for below code block: https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_small_1024
    input_ids = translate_tokenizer("translate to zh: " + text, return_tensors="pt")
    generated_tokens = translate_model.generate(**input_ids.to('cuda'))
    return translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) 


# Preparing Summariser
# Below Code Line From: https://huggingface.co/PeterBanning71/long-t5-tfg?library=transformers
summariser = pipeline("summarization", model="PeterBanning71/long-t5-tfg")

def summarise_text(text):
    # Below Code Line From: https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary, https://huggingface.co/facebook/bart-large-cnn
    return summariser(text)[0]["summary_text"]

nltk.download('punkt_tab')

# Ranked Retrieval
# Below code block from: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls
rr_tokenizer = AutoTokenizer.from_pretrained("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", padding_side="left")
rr_model = AutoModelForSequenceClassification.from_pretrained("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", torch_dtype=torch.float16).eval()

# Below code block from: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls
def format_instruction(instruction, query, doc):
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    output = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
    return output

# Code in below function from: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls
def ranked_retrieve(context, question):
    max_length = 128 * 1024
    task = "Given a web search query, retrieve relevant passages that answer the query"
    context_sentences = nltk.tokenize.sent_tokenize(context, language='english')
    queries = [question for _ in range(len(context_sentences))]

    pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, context_sentences)]
    
    inputs = rr_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    logits = rr_model(**inputs).logits.squeeze()
    relevancies = logits.sigmoid()
    relevancies = relevancies.tolist()
    mean_relevancy = np.mean(relevancies)

    return_context = ""

    for i in range(len(relevancies)):
        if relevancies[i] >= mean_relevancy:
            return_context += context_sentences[i] + " "

    return_context = return_context[:len(return_context) - 1]
    return return_context


'''
References

1. Bird, Steven, Edward Loper and Ewan Klein (2009).
Natural Language Processing with Python.  O'Reilly Media Inc.

2. Q. Team, “Qwen3-embedding,” May 2025. [Online]. Available: https://qwenlm.github.io/blog/qwen3/

3. T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz,
J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger,
M. Drame, Q. Lhoest, and A. M. Rush, “Transformers: State-of-the-art natural language processing,”
in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations. Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45. [Online]. Available:
https://www.aclweb.org/anthology/2020.emnlp-demos.6

'''