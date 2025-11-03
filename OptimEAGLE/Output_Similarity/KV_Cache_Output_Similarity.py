from FlagEmbedding import BGEM3FlagModel
import numpy as np

# Code in main() From: https://huggingface.co/BAAI/bge-m3
def main():
    print("\n\n*******************************\nStarting KV_Cache_Output_Similarity.py\n\n")
    
    model = BGEM3FlagModel('BAAI/bge-m3',  
                        use_fp16=True)
    
    # Analysing KV Cache Test Outputs
    # Note: Copy Experimental Output Into Corresponding Empty Arrays
    baseline = []
    bottom_5 = []
    top_5 = []
    bottom_third = []
    middle_third = []
    top_third = []
    similarity_scores = []
    
    # Output Quality of BOTTOM-5 Test
    for i in range(len(baseline)):
        embeddings_1 = model.encode(baseline[i])['dense_vecs']
        embeddings_2 = model.encode(bottom_5[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("\nMean Similarity Score Between BASELINE-0 & BOTTOM-5: ", np.mean(similarity_scores))

    # Output Quality of TOP-5 Test
    similarity_scores = []
    
    for i in range(len(baseline)):
        embeddings_1 = model.encode(baseline[i])['dense_vecs']
        embeddings_2 = model.encode(top_5[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("\nMean Similarity Score Between BASELINE-0 & TOP-5: ", np.mean(similarity_scores))

    # Output Quality of BOTTOM-THIRD
    similarity_scores = []
    
    for i in range(len(baseline)):
        embeddings_1 = model.encode(baseline[i])['dense_vecs']
        embeddings_2 = model.encode(bottom_third[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("\nMean Similarity Score Between BASELINE-0 & BOTTOM-THIRD: ", np.mean(similarity_scores))

    # Output Quality of MIDDLE-THIRD
    similarity_scores = []
    
    for i in range(len(baseline)):
        embeddings_1 = model.encode(baseline[i])['dense_vecs']
        embeddings_2 = model.encode(middle_third[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("\nMean Similarity Score Between BASELINE-0 & MIDDLE-THIRD: ", np.mean(similarity_scores))

    # Output Quality of TOP-THIRD
    similarity_scores = []
    
    for i in range(len(baseline)):
        embeddings_1 = model.encode(baseline[i])['dense_vecs']
        embeddings_2 = model.encode(top_third[i])['dense_vecs']
        similarity = embeddings_1 @ embeddings_2.T
        similarity_scores.append(similarity)

    print("\nMean Similarity Score Between BASELINE-0 & TOP-THIRD: ", np.mean(similarity_scores))

    print("\n\n*******************************\nFinished Running KV_Cache_Output_Similarity.py\n\n")

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