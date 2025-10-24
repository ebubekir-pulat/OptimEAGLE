print("\n\n*******************************\nStarting LBE_Output_Similarity.py\n\n")

from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Code in main() From: https://huggingface.co/BAAI/bge-m3
def main():
    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    # Comparing Autoregressive Decoding Outputs
    # Note: Copy Experimental Output Into Corresponding Empty Arrays
    autoreg_standard = []
    autoreg_summ = []
    autoreg_skip = []
    autoreg_ranked = []
    autoreg_similarity_scores = []
    
    # Output Quality of AutoReg-summ
    for i in range(len(autoreg_standard)):
        embeddings_1 = model.encode(autoreg_standard[i])['dense_vecs']
        embeddings_2 = model.encode(autoreg_summ[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        autoreg_similarity_scores.append(similarity)

    print("\nMean Similarity Score Between AutoReg-Standard & AutoReg-Summ: ", np.mean(autoreg_similarity_scores))

    # Output Quality of AutoReg-skip
    autoreg_similarity_scores = []
    
    for i in range(len(autoreg_standard)):
        embeddings_1 = model.encode(autoreg_standard[i])['dense_vecs']
        embeddings_2 = model.encode(autoreg_skip[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        autoreg_similarity_scores.append(similarity)

    print("\nMean Similarity Score Between AutoReg-Standard & AutoReg-skip: ", np.mean(autoreg_similarity_scores))

    # Output Quality of AutoReg-ranked
    autoreg_similarity_scores = []
    
    for i in range(len(autoreg_standard)):
        embeddings_1 = model.encode(autoreg_standard[i])['dense_vecs']
        embeddings_2 = model.encode(autoreg_ranked[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        autoreg_similarity_scores.append(similarity)

    print("\nMean Similarity Score Between AutoReg-Standard & AutoReg-ranked: ", np.mean(autoreg_similarity_scores))

    print("\n\n*******************************\nFinished Running LBE_Output_Similarity.py\n\n")

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