print("\n\n*******************************\nStarting AAI_EAGLE3_AutoReg_Output_Sim.py\n\n")

from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Code in main() From: https://huggingface.co/BAAI/bge-m3
def main():
    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    # Comparing EAGLE-3 Output on Original AAI Texts vs Autoregressive Output on Original AAI Texts

    # Note: Copy Experimental Output Into Corresponding Empty Arrays
    autoreg_notrans = []
    eagle3_notrans = []
    aai_similarity_scores = []
    
    for i in range(len(autoreg_notrans)):
        embeddings_1 = model.encode(autoreg_notrans[i])['dense_vecs']
        embeddings_2 = model.encode(eagle3_notrans[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        aai_similarity_scores.append(similarity)

    print("\nMean Similarity Score Between AutoReg-NoTrans & EAGLE3-NoTrans: ", np.mean(aai_similarity_scores))

    print("\n\n*******************************\nFinished Running AAI_EAGLE3_AutoReg_Output_Sim.py\n\n")

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