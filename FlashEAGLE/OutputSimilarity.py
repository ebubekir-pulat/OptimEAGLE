# pip install -U FlagEmbedding

from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Below Code From: https://huggingface.co/BAAI/bge-m3

if __name__ == '__main__':
    compresseagle_answers = []
    flasheagle_answers = []
    similarity_scores = []

    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    for i in range(len(compresseagle_answers)):
        embeddings_1 = model.encode(compresseagle_answers[i])['dense_vecs']
        embeddings_2 = model.encode(flasheagle_answers[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("Mean Similarity Score: ", np.mean(similarity_scores))

# Note: Put Reference For Above Link

'''
References

1.

'''