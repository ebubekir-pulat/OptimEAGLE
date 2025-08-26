print("\n\n*******************************\nStarting Output_Similarity.py\n\n")

# pip install -U FlagEmbedding

from FlagEmbedding import BGEM3FlagModel
import numpy as np
import pandas as pd
import sys

# Below Code From: https://huggingface.co/BAAI/bge-m3
def main(file1, file2):
    first_answers = pd.read_json(file1, lines=True)
    second_answers = pd.read_json(file2, lines=True)
    similarity_scores = []

    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    for i in range(len(first_answers)):
        embeddings_1 = model.encode(first_answers[i]["output"])['dense_vecs']
        embeddings_2 = model.encode(second_answers[i]["output"])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print("Mean Similarity Score: ", np.mean(similarity_scores))

# Note: Put Reference For Above Link

if __name__ == '__main__':
    main(sys.argv[0], sys.argv[1])


print("\n\n*******************************\nFinished Running Output_Similarity.py\n\n")

'''
References

1.

'''