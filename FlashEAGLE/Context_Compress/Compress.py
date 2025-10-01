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

# Sentence Retrieval
def sentence_retrieve(context):
    context_sentences = nltk.tokenize.sent_tokenize(context, language='english')
    return_context = ""

    for i in range(len(context_sentences)):
        if i % 2 != 0:
            return_context += context_sentences[i] + " "

    return_context = return_context[:len(return_context) - 1]
    return return_context

# Reference for below code block: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
rr_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
rr_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
rr_model.eval()

def ranked_retrieve(context, input):
    context_sentences = nltk.tokenize.sent_tokenize(context, language='english')
    relevancies = []

    for sentence in context_sentences:
        # Reference for below code block: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
        features = rr_tokenizer([input], [sentence],  padding=True, truncation=True, return_tensors="pt")    
        with torch.no_grad():
            score = rr_model(**features).logits
            print(score)
            relevancies.append(score[0])

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

2. T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz,
J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger,
M. Drame, Q. Lhoest, and A. M. Rush, “Transformers: State-of-the-art natural language processing,”
in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations. Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45. [Online]. Available:
https://www.aclweb.org/anthology/2020.emnlp-demos.6

'''