print("\n\n*******************************\nStarting Output_Similarity.py\n\n")

from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Below Code From: https://huggingface.co/BAAI/bge-m3
def main():
    first_answers = ["My name is Bob.", "Where is the Chicken?"]
    second_answers = ["My name is Bob.", "My name is Bob."]
    similarity_scores = []

    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    for i in range(len(first_answers)):
        embeddings_1 = model.encode(first_answers[i])['dense_vecs']
        embeddings_2 = model.encode(second_answers[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("Mean Similarity Score: ", np.mean(similarity_scores))
    print("\n\n*******************************\nFinished Running Output_Similarity.py\n\n")

if __name__ == '__main__':
    main()

'''
References

1. J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “Bge m3-embedding: Multi-lingual, multi-functionality,
multi-granularity text embeddings through self-knowledge distillation,” 2023.

2. S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff, “C-pack: Packaged resources to advance general chinese embed-
ding,” 2023.

3. S. Xiao, Z. Liu, P. Zhang, and X. Xing, “Lm-cocktail: Resilient tuning of language models via model merging,”
2023.

4. P. Zhang, S. Xiao, Z. Liu, Z. Dou, and J.-Y. Nie, “Retrieve anything to augment large language models,” 2023.

'''