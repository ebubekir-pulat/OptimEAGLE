# pip install -U FlagEmbedding

from FlagEmbedding import BGEM3FlagModel
import numpy as np
import pandas as pd

# Below Code From: https://huggingface.co/BAAI/bge-m3

if __name__ == '__main__':
    file_name = 'AAI_output'
    compresseagle_answers = pd.read_json(f"{file_name}_ceagle.jsonl", lines=True)
    flasheagle_answers = pd.read_json(f"{file_name}_feagle.jsonl", lines=True)
    similarity_scores = []

    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    for i in range(len(compresseagle_answers)):
        embeddings_1 = model.encode(compresseagle_answers[i]["output"])['dense_vecs']
        embeddings_2 = model.encode(flasheagle_answers[i]["output"])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("Mean Similarity Score: ", np.mean(similarity_scores))

# Note: Put Reference For Above Link

'''
References

1.

'''