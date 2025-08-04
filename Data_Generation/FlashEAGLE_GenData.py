print("\n\n*******************************\nStarting FlashEAGLE_GenData.py\n\n")

import pandas as pd    
import numpy as np
import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template

# Getting Spec-Bench Questions
# Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
jsonObj = pd.read_json(path_or_buf='../question.jsonl', lines=True)
sb_prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]

base_model_paths = ["lmsys/vicuna-13b-v1.3",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.3-70B-Instruct"]

EAGLE_model_paths = ["yuhuili/EAGLE3-Vicuna1.3-13B",
                     "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
                     "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                     "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"]

def template_getter(model_index):
    if model_index == 0:
        return "vicuna"
    else:
        return base_model_paths[model_index]

def model_init(model_index):
    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[model_index],
        ea_model_path=EAGLE_model_paths[model_index],
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model

# Preparing for assessment
models_to_test = [2, 3]
max_new_tokens = 1024
temp = 0.0

print("\nGeneration Settings Chosen:")
print("Max New Tokens: ", max_new_tokens)
print("Temperature: ", temp, "\n")

# Generation Loop
for model_index in models_to_test:
    model = model_init(model_index)

    for i in range(len(sb_prompts)):
        for question in sb_prompts[i]:
            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            your_message = question
            conv = get_conversation_template(template_getter(model_index))
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()

            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            output_ids = model.eagenerate(input_ids, temperature=temp, max_new_tokens=len(prompt)+max_new_tokens, log=True)
            generated_data=model.tokenizer.decode(output_ids[0][0])

            response_index = generated_data.find("### Assistant: ", 300) + len("### Assistant: ")
            generated_data = generated_data[response_index:]
            generated_data = generated_data[:generated_data.find("### Human:")]
            generated_data = generated_data.strip()
            
            print("\n\n*********************************************\nPrompt: ", question)
            print("\nResponse: ", generated_data)


'''
References

1. H. Xia, Z. Yang, Q. Dong, P. Wang, Y. Li, T. Ge, T. Liu, W. Li, and Z. Sui, “Unlocking efficiency in large
language model inference: A comprehensive survey of speculative decoding,” in Findings of the Association
for Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok,
Thailand: Association for Computational Linguistics, Aug. 2024, pp. 7655–7671. [Online]. Available:
https://aclanthology.org/2024.findings-acl.456/

2. Y. Li, F. Wei, C. Zhang, and H. Zhang, “Eagle-3: Scaling up inference acceleration of large language models
via training-time test,” 2025. [Online]. Available: https://arxiv.org/abs/2503.01840

3. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE: Speculative sampling requires rethinking feature
uncertainty,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings
of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett,
and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 28 935–28 948. [Online]. Available:
https://proceedings.mlr.press/v235/li24bt.html

4. T. Dao, “FlashAttention-2: Faster attention with better parallelism and work partitioning,” in International
Conference on Learning Representations (ICLR), 2024.

5. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang,
J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Proceedings of
the 37th International Conference on Neural Information Processing Systems, ser. NIPS ’23. Red Hook, NY,
USA: Curran Associates Inc., 2023.

6. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, “Flashattention: fast and memory-efficient exact attention
with io-awareness,” in Proceedings of the 36th International Conference on Neural Information Processing
Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

7. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE-2: Faster inference of language models with dynamic
draft trees,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics,
Nov. 2024, pp. 7421–7432. [Online]. Available: https://aclanthology.org/2024.emnlp-main.422/

8. W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez,
I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” March
2023. [Online]. Available: https://lmsys.org/blog/2023-03-30-vicuna/

9. A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten,
A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev,
A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern,
C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer,
C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan,
D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith,
F. Radenovic, F. Guzm´an, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Thattai, G. Nail, G. Mialon,
G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra,
I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. van der
Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak,
J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li,
K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. Rantala-Yeary, L. van der
Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. de Oliveira,
M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova,
M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev,
N. Chatterji, N. Zhang, O. Duchenne, O. C¸elebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava,
P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S.
Cabral, R. Stojnic, R. Raileanu, R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly,
R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov,
S. Nie, S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman,
S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom,
T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet,
V. Do, V. Vogeti, V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang,
X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song,
Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain,
A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Baevski,
A. Feinstein, A. Kallet, A. Sangani, A. Teo, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho,
A. Poulton, A. Ryan, A. Ramchandani, A. Dong, A. Franco, A. Goyal, A. Saraf, A. Chowdhury, A. Gabriel,
A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer, B. Leonhardi, B. Huang, B. Loyd, B. D. Paola,
B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti, B. Spence, B. Stojkovic, B. Gamido, B. Montalvo,
C. Parker, C. Burton, C. Mejia, C. Liu, C. Wang, C. Kim, C. Zhou, C. Hu, C.-H. Chu, C. Cai, C. Tindal,
C. Feichtenhofer, C. Gao, D. Civin, D. Beaty, D. Kreymer, D. Li, D. Adkins, D. Xu, D. Testuggine, D. David,
D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil, E. Montgomery, E. Presani,
E. Hahn, E. Wood, E.-T. Le, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian,
F. Kokkinos, F. Ozgenel, F. Caggioni, F. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee,
G. Halpern, G. Herman, G. Sizov, Guangyi, Zhang, G. Lakshminarayanan, H. Inan, H. Shojanazeri, H. Zou,
H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk, H. Aspegren, H. Goldman, H. Zhan, I. Damlaj, I. Molybog,
I. Tufanov, I. Leontiadis, I.-E. Veliche, I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Lam, J. Asher, J.-B. Gaya,
J. Marcus, J. Tang, J. Chan, J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin, J. Yang, J. Cummings,
J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, K. H. U, K. Saxena, K. Khandelwal,
K. Zand, K. Matosich, K. Veeraraghavan, K. Michelena, K. Li, K. Jagadeesh, K. Huang, K. Chawla, K. Huang,
L. Chen, L. Garg, L. A, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa,
M. Avalani, M. Bhatt, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi,
M. Keneally, M. Liu, M. L. Seltzer, M. Valko, M. Restrepo, M. Patel, M. Vyatskov, M. Samvelyan, M. Clark,
M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal, N. Santhanam, N. Parks, N. White,
N. Bawa, N. Singhal, N. Egebo, N. Usunier, N. Mehta, N. P. Laptev, N. Dong, N. Cheng, O. Chernoguz, O. Hart,
O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab, P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollar,
P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang, R. Alao, R. Rodriguez, R. Ayub, R. Murthy, R. Nayani,
R. Mitra, R. Parthasarathy, R. Li, R. Hogan, R. Battey, R. Wang, R. Howes, R. Rinott, S. Mehta, S. Siby, S. J.
Bondu, S. Datta, S. Chugh, S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Mahajan, S. Verma, S. Yamamoto,
S. Ramaswamy, S. Lindsay, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Patil, S. Shankar, S. Zhang, S. Zhang,
S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad,
S. Gupta, S. Deng, S. Cho, S. Virk, S. Subramanian, S. Choudhury, S. Goldman, T. Remez, T. Glaser, T. Best,
T. Koehler, T. Robinson, T. Li, T. Zhang, T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi,
V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. Poenaru, V. T. Mihailescu, V. Ivanov, W. Li,
W. Wang, W. Jiang, W. Bouaziz, W. Constable, X. Tang, X. Wu, X. Wang, X. Wu, X. Gao, Y. Kleinman,
Y. Chen, Y. Hu, Y. Jia, Y. Qi, Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Yu, Wang, Y. Zhao, Y. Hao, Y. Qian,
Y. Li, Y. He, Z. Rait, Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, Z. Zhao, and Z. Ma, “The llama 3 herd of
models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21783

10. DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025. [Online].
Available: https://arxiv.org/abs/2501.12948

'''

print("\n\n*******************************\nFinished Running FlashEAGLE_GenData.py\n\n")