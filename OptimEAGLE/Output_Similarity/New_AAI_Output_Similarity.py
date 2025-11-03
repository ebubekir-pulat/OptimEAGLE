print("\n\n*******************************\nStarting New_AAI_Output_Similarity.py\n\n")

from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Code in main() From: https://huggingface.co/BAAI/bge-m3
def main():
    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    # Comparing Autoregressive Decoding Outputs

    # Note: Copy Experimental Output Into Corresponding Empty Arrays
    autoreg_notrans = []
    autoreg_trans = []
    autoreg_similarity_scores = []
    
    for i in range(len(autoreg_notrans)):
        embeddings_1 = model.encode(autoreg_notrans[i])['dense_vecs']
        embeddings_2 = model.encode(autoreg_trans[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        autoreg_similarity_scores.append(similarity)

    print("\nMean Similarity Score Between AutoReg-NoTranslate & AutoReg-Translate: ", np.mean(autoreg_similarity_scores))

    # Comparing EAGLE-3 Output

    # Note: Copy Experimental Output Into Corresponding Empty Array
    eagle3_trans = []
    eagle3_similarity_scores = []
    
    for i in range(len(autoreg_notrans)):
        embeddings_1 = model.encode(autoreg_notrans[i])['dense_vecs']
        embeddings_2 = model.encode(eagle3_trans[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        eagle3_similarity_scores.append(similarity)

    print("\nMean Similarity Score Between AutoReg-NoTranslate & EAGLE3-Translate: ", np.mean(eagle3_similarity_scores))

    print("\n\n*******************************\nFinished Running New_AAI_Output_Similarity.py\n\n")

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