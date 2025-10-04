# Task-Specific LIO Testing on Spec-Bench 
# Hyperparameters: test_runs, max_new_tokens, temp

import subprocess

subprocess.run(
    ["sudo", "apt-get", "-y", "install", "libnuma-dev"], check=True
)

subprocess.run(
    ["pip", "install", "uv"], check=True
)

subprocess.run(
    ["uv", "pip", "install", "sglang[all]>=0.5.3rc0"], check=True
)

subprocess.run(
    ["pip", "install", "numpy==1.26.4"], check=True
)

subprocess.run(
    ["nvidia-smi"], check=True
)

subprocess.run(
    ["hf", "auth", "login", "--token", "token"], check=True
)

import time
import numpy as np
import json
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, terminate_process
import openai
import Data
import hashlib

def main():
    print("\n\n*******************************\nStarting LIO_TaskSpecific_Test.py\n\n")

    print("Python Version:")

    subprocess.run(
        ["python", "--version"], check=True
    )

    LIO_model_paths = ["openai/gpt-oss-20b"]
    base_model_paths = ["meta-llama/Llama-3.1-8B-Instruct"]
    EAGLE_model_paths = ["jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"]

    prompts = Data.specbench()
    print("Spec-Bench Dataset Shape: ", np.shape(prompts))

    tasks = Data.specbench_tasks()
    print("Spec-Bench Tasks Shape: ", np.shape(tasks))

    tasks_set = {tasks}
    optim_params = {}

    # Preparing LIO SGLANG
    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html, https://docs.sglang.ai/basic_usage/send_request.html
    server_process, port = launch_server_cmd(
        f"""
    python3 -m sglang.launch_server --model-path {LIO_model_paths[0]} --dtype float16
    """
    )

    # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    wait_for_server(f"http://localhost:{port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    for task in tasks_set:
        LIO_prompt = f"Generate optimal hyperparameters for EAGLE-3 speculative decoding with SGLANG, where the \
                    base model to be used is {base_model_paths[0]}, the EAGLE-3 model to be used is {EAGLE_model_paths[0]} and \
                    the dataset to be tested on is Spec-Bench. Spec-Bench is a benchmark covering multi-turn conversation, \
                    translation, summarisation, question answering, mathematical reasoning and retrieval-augmented generation, \
                    consisting of samples from the MT-bench, WMT14 DE-EN, CNN/Daily Mail, Natural Questions, GSM8K and DPR \
                    datasets. The specific task type to optimise for is {task}. Choose hyperparameters that optimise acceptance length, tokens generated per second and \
                    wall-time speedup. Provide values for these parameters, or ignore them if intend to keep default: --kv-cache-dtype ('auto', 'fp8_e5m2', 'fp8_e4m3'), \
                    --stream-interval, --max-prefill-tokens, --chunked-prefill-size, --speculative-num-steps, \
                    --disable-outlines-disk-cache (To Select Option, Simply Write Parameter), --enable-tokenizer-batch-encode (To Select Option, Simply Write Parameter), \
                    --speculative-eagle-topk, --speculative-num-draft-tokens, --speculative-attention-mode (prefill or decode), \
                    --disable-radix-cache (To Select Option, Simply Write Parameter), --cuda-graph-max-bs, \
                    --disable-custom-all-reduce (To Select Option, Simply Write Parameter), --disable-overlap-schedule (To Select Option, Simply Write Parameter), \
                    --enable-mixed-chunk (To Select Option, Simply Write Parameter), --enable-dp-attention (To Select Option, Simply Write Parameter), \
                    --enable-dp-lm-head (To Select Option, Simply Write Parameter), --enable-two-batch-overlap (To Select Option, Simply Write Parameter), \
                    --tbo-token-distribution-threshold, --enable-torch-compile (To Select Option, Simply Write Parameter), --torch-compile-max-bs, \
                    --triton-attention-reduce-in-fp32 (To Select Option, Simply Write Parameter), --triton-attention-num-kv-splits, --num-continuous-decode-steps, \
                    --enable-memory-saver (To Select Option, Simply Write Parameter), --allow-auto-truncate (To Select Option, Simply Write Parameter), \
                    --delete-ckpt-after-loading (To Select Option, Simply Write Parameter), --enable-return-hidden-states (To Select Option, Simply Write Parameter), \
                    --disable-fast-image-processor (To Select Option, Simply Write Parameter), --disable-chunked-prefix-cache (To Select Option, Simply Write Parameter), \
                    --flashinfer-mla-disable-ragged (To Select Option, Simply Write Parameter) and --disable-shared-experts-fusion (To Select Option, Simply Write Parameter). \
                    Generate the hyperparameters in the format, --dtype ('auto', 'half', 'bfloat16', 'float', 'float32'), --parameter-name value, and do that \
                    for all hyperparameters. For parameters where I have specified, 'To Select Option, Simply Write Parameter', don't include a value if you want to \
                    use that setting, and don't write the parameter at all if you don't want that setting. Before providing the hyperparameters, \
                    put a #START delimiter, and when finished, put a #END delimiter. THIS IS IMPORTANT. Note, if choosing to keep the default value for a parameter, \
                    DO NOT LIST THE PARAMETER IN BETWEEN THE DELIMITERS. Make sure to follow the format, and ensure your total output is within 4096 tokens MAXIMUM! \
                    REMEMBER TO OPTIMISE FOR THE {task} TASK TYPE SPECIFICALLY!"

        # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
        response = client.chat.completions.create(
            model=LIO_model_paths[0],
            messages=[
                {"role": "user", "content": LIO_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )

        # Reference for below code line: https://stackoverflow.com/questions/77444332/openai-python-package-error-chatcompletion-object-is-not-subscriptable 
        LIO_output = response.choices[0].message.content
        LIO_output = Data.extract_LIO_response(LIO_output)
        print("LIO Output: ", LIO_output)
        optim_params[task] = LIO_output

    # Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
    terminate_process(server_process)

    servers = {}

    for task in optim_params:
        # Preparing SGLANG with EAGLE3
        # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
        server_process, port = launch_server_cmd(
            f"""
        python3 -m sglang.launch_server --model {base_model_paths[0]}  --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path {EAGLE_model_paths[0]} {optim_params[task]}
        """
        )

        # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
        wait_for_server(f"http://localhost:{port}")
        client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

        servers[task] = [server_process, port, client]


    LIO_outputs = []
    # Hyperparameters
    test_runs = 3
    max_new_tokens = 1024
    temp = 0.0

    print("\nEvaluation Settings Chosen:")
    print("Dataset: Spec-Bench")
    print("Test Runs: ", test_runs)
    print("Max New Tokens: ", max_new_tokens)
    print("Temperature: ", temp, "\n")

    # LIO Assessment Loop
    wall_times = []
    token_rates = []
    input_tokens = []
    output_tokens = []

    for test_run in range(test_runs):
        run = 1
        for i in range(len(prompts)):
            print("Test Run: ", test_run)
            print("Test Question: ", run)
            run += 1

            prompt = prompts[i][0]
            
            server_process, port, client = servers[tasks[i]]
            start = time.perf_counter_ns()

            # Below Code Block From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
            response = client.chat.completions.create(
                model=base_model_paths[0],
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_new_tokens,
            )

            finish = time.perf_counter_ns()

            # Reference for below code line: https://stackoverflow.com/questions/77444332/openai-python-package-error-chatcompletion-object-is-not-subscriptable 
            model_output = response.choices[0].message.content
            
            elapsed = finish - start
            wall_times.append(elapsed)

            new_tokens = response.usage.completion_tokens
            tokens_per_second = new_tokens / (elapsed * pow(10, -9))
            token_rates.append(tokens_per_second)
            output_tokens.append(new_tokens)

            input_tokens.append(response.usage.prompt_tokens)

            # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
            output = {
                "id": hashlib.md5((str(test_run) + prompt + model_output).encode()).hexdigest(),
                "output": model_output
            }
            LIO_outputs.append(output)

    # Print LIO Results
    print(f"LIO Results for {LIO_model_paths[0]}:")
    print(f"Dataset: Spec-Bench")
    print(f"EAGLE Model: {EAGLE_model_paths[0]}")
    print(f"Base Model: {base_model_paths[0]}")
    print("Mean Wall Time (ns): ", np.mean(wall_times))
    print("Mean Tokens Generated/s: ", np.mean(token_rates))

    for task in servers:
        server_process, port, client = servers[task]

        # Below Code Line From: https://docs.sglang.ai/advanced_features/speculative_decoding.html
        terminate_process(server_process)

    output_name = f"LIO_TaskSpecific_Output_{LIO_model_paths[0].replace('/', '-')}_{EAGLE_model_paths[0].replace('/', '-')}.jsonl" 
    
    # Below Code Block From: https://github.com/sgl-project/SpecForge/blob/main/scripts/prepare_data.py
    with open(output_name, "x") as f:
        for output in LIO_outputs:
            f.write(json.dumps(output) + "\n")

    print("Walltimes Array: ", wall_times)
    print("Input Tokens Array: ", input_tokens)
    print("Output Tokens Array: ", output_tokens)
    print("Tokens Generated Per Second Array: ", token_rates)

    print("\n\nOutput Data: \n")

    for output in LIO_outputs:
        print(output)

    subprocess.run(
        ["hf", "auth", "logout", "--token", "token"], check=True
    )

    print("\n\n*******************************\nFinished Running LIO_TaskSpecific_Test.py\n\n")

if __name__ == "__main__":
    main()

''' 
References

1. A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten,
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

2. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE: Speculative sampling requires rethinking feature
uncertainty,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings
of Machine Learning Research, R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett,
and F. Berkenkamp, Eds., vol. 235. PMLR, 21–27 Jul 2024, pp. 28 935–28 948. [Online]. Available:
https://proceedings.mlr.press/v235/li24bt.html

3. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE-2: Faster inference of language models with dynamic
draft trees,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational
Linguistics, Nov. 2024, pp. 7421–7432. [Online]. Available: https://aclanthology.org/2024.emnlp-main.422/

4. Y. Li, F. Wei, C. Zhang, and H. Zhang, “Eagle-3: Scaling up inference acceleration of large language models
via training-time test,” 2025. [Online]. Available: https://arxiv.org/abs/2503.01840

5. C. W. F. Y. S. S. Y. W. Y. Z. Y. H. H. Z. Y. Z. Shenggui Li, Yikai Zhu, “Specforge: Train speculative decoding
models effortlessly,” https://github.com/sgl-project/specforge, 2025.

6. OpenAI, “gpt-oss-120b gpt-oss-20b model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925

'''