# pip install -U FlagEmbedding

from FlagEmbedding import BGEM3FlagModel

# Below Code From: https://huggingface.co/BAAI/bge-m3

eagle_answers = ["What is BGE M3?", "Defination of BM25"]
approx_answers = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True)

embeddings_1 = model.encode(eagle_answers, 
                            batch_size=12, 
                            max_length=4096,
                            )['dense_vecs']
embeddings_2 = model.encode(approx_answers)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)


# Note: Put Reference For Above Link