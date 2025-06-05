import pandas as pd    
import time
import numpy as np
import torch
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template

# Getting Spec-Bench Questions
# Below line from: https://stackoverflow.com/questions/50475635/loading-jsonl-file-as-json-objects
jsonObj = pd.read_json(path_or_buf='../question.jsonl', lines=True)
prompts = [jsonObj.at[i, 'turns'] for i in range(len(jsonObj))]

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
    elif model_index in [2, 3]:
        return "llama-3-chat"
    else:
        return base_model_paths[model_index]

def model_init(model_index):
    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[model_index],
        ea_model_path=EAGLE_model_paths[model_index],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
        # offload_folder="offload" # Code Line From: https://github.com/nomic-ai/gpt4all/issues/239
    )

    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model

# Preparing for assessment
wall_times = []
token_rates = []
models_to_test = [0, 1, 2]
test_runs = 1

# Assessment Loop
for model_index in models_to_test:
    model = model_init(model_index)
    for test_run in range(test_runs):
        run = 1
        for i in range(160, 480):
            print("Test Run: ", test_run)
            print("SB Question: ", run)
            run += 1

            # Below Code Block From: https://github.com/SafeAILab/EAGLE
            your_message = prompts[i]
            if len(your_message) == 1: 
                your_message = your_message[0]
            else: 
                raise("Message Length Above 1")
            conv = get_conversation_template(template_getter(model_index))
            conv.append_message(conv.roles[0], your_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = model.tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()

            start = time.perf_counter_ns()

            # Below Code Line From: https://github.com/SafeAILab/EAGLE
            output_ids = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=256, log=True)
            #output=model.tokenizer.decode(output_ids[0])

            finish = time.perf_counter_ns()
            elapsed = finish - start
            wall_times.append(elapsed)

            num_tokens = int(output_ids[1])
            tokens_per_second = num_tokens / (elapsed * pow(10, -9))
            token_rates.append(tokens_per_second)

# Print Results
print("Results:")
print("Mean Wall Time (ns): ", np.mean(wall_times))
print("Mean Tokens/s: ", np.mean(token_rates))

'''
References

1. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, 'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty', in International Conference on Machine Learning.

2. Li, Y, Wei, F, Zhang, C & Zhang, H 2024, 'EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees', in Empirical Methods in Natural Language Processing.

3. Li, Y, Wei, F, Zhang, C & Zhang, H 2025, 'EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test', <https://arxiv.org/abs/2503.01840>.

4. Zheng, L, Chiang, WL, Sheng, Y, Zhuang, S, Wu, Z, Zhuang, Y, Lin, Z, Li, Z, Li, D, Xing, EP, Zhang, H, Gonzalez, JE & Stoica, I 2023, 'Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena', <https://arxiv.org/abs/2306.05685>.


@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}
@misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
      title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
      author={DeepSeek-AI},
      year={2025},
      eprint={2501.12948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.12948}, 
}
@misc{grattafiori2024llama3herdmodels,
      title={The Llama 3 Herd of Models}, 
      author={Aaron Grattafiori and Abhimanyu Dubey and Abhinav Jauhri and Abhinav Pandey and Abhishek Kadian and Ahmad Al-Dahle and Aiesha Letman and Akhil Mathur and Alan Schelten and Alex Vaughan and Amy Yang and Angela Fan and Anirudh Goyal and Anthony Hartshorn and Aobo Yang and Archi Mitra and Archie Sravankumar and Artem Korenev and Arthur Hinsvark and Arun Rao and Aston Zhang and Aurelien Rodriguez and Austen Gregerson and Ava Spataru and Baptiste Roziere and Bethany Biron and Binh Tang and Bobbie Chern and Charlotte Caucheteux and Chaya Nayak and Chloe Bi and Chris Marra and Chris McConnell and Christian Keller and Christophe Touret and Chunyang Wu and Corinne Wong and Cristian Canton Ferrer and Cyrus Nikolaidis and Damien Allonsius and Daniel Song and Danielle Pintz and Danny Livshits and Danny Wyatt and David Esiobu and Dhruv Choudhary and Dhruv Mahajan and Diego Garcia-Olano and Diego Perino and Dieuwke Hupkes and Egor Lakomkin and Ehab AlBadawy and Elina Lobanova and Emily Dinan and Eric Michael Smith and Filip Radenovic and Francisco Guzmán and Frank Zhang and Gabriel Synnaeve and Gabrielle Lee and Georgia Lewis Anderson and Govind Thattai and Graeme Nail and Gregoire Mialon and Guan Pang and Guillem Cucurell and Hailey Nguyen and Hannah Korevaar and Hu Xu and Hugo Touvron and Iliyan Zarov and Imanol Arrieta Ibarra and Isabel Kloumann and Ishan Misra and Ivan Evtimov and Jack Zhang and Jade Copet and Jaewon Lee and Jan Geffert and Jana Vranes and Jason Park and Jay Mahadeokar and Jeet Shah and Jelmer van der Linde and Jennifer Billock and Jenny Hong and Jenya Lee and Jeremy Fu and Jianfeng Chi and Jianyu Huang and Jiawen Liu and Jie Wang and Jiecao Yu and Joanna Bitton and Joe Spisak and Jongsoo Park and Joseph Rocca and Joshua Johnstun and Joshua Saxe and Junteng Jia and Kalyan Vasuden Alwala and Karthik Prasad and Kartikeya Upasani and Kate Plawiak and Ke Li and Kenneth Heafield and Kevin Stone and Khalid El-Arini and Krithika Iyer and Kshitiz Malik and Kuenley Chiu and Kunal Bhalla and Kushal Lakhotia and Lauren Rantala-Yeary and Laurens van der Maaten and Lawrence Chen and Liang Tan and Liz Jenkins and Louis Martin and Lovish Madaan and Lubo Malo and Lukas Blecher and Lukas Landzaat and Luke de Oliveira and Madeline Muzzi and Mahesh Pasupuleti and Mannat Singh and Manohar Paluri and Marcin Kardas and Maria Tsimpoukelli and Mathew Oldham and Mathieu Rita and Maya Pavlova and Melanie Kambadur and Mike Lewis and Min Si and Mitesh Kumar Singh and Mona Hassan and Naman Goyal and Narjes Torabi and Nikolay Bashlykov and Nikolay Bogoychev and Niladri Chatterji and Ning Zhang and Olivier Duchenne and Onur Çelebi and Patrick Alrassy and Pengchuan Zhang and Pengwei Li and Petar Vasic and Peter Weng and Prajjwal Bhargava and Pratik Dubal and Praveen Krishnan and Punit Singh Koura and Puxin Xu and Qing He and Qingxiao Dong and Ragavan Srinivasan and Raj Ganapathy and Ramon Calderer and Ricardo Silveira Cabral and Robert Stojnic and Roberta Raileanu and Rohan Maheswari and Rohit Girdhar and Rohit Patel and Romain Sauvestre and Ronnie Polidoro and Roshan Sumbaly and Ross Taylor and Ruan Silva and Rui Hou and Rui Wang and Saghar Hosseini and Sahana Chennabasappa and Sanjay Singh and Sean Bell and Seohyun Sonia Kim and Sergey Edunov and Shaoliang Nie and Sharan Narang and Sharath Raparthy and Sheng Shen and Shengye Wan and Shruti Bhosale and Shun Zhang and Simon Vandenhende and Soumya Batra and Spencer Whitman and Sten Sootla and Stephane Collot and Suchin Gururangan and Sydney Borodinsky and Tamar Herman and Tara Fowler and Tarek Sheasha and Thomas Georgiou and Thomas Scialom and Tobias Speckbacher and Todor Mihaylov and Tong Xiao and Ujjwal Karn and Vedanuj Goswami and Vibhor Gupta and Vignesh Ramanathan and Viktor Kerkez and Vincent Gonguet and Virginie Do and Vish Vogeti and Vítor Albiero and Vladan Petrovic and Weiwei Chu and Wenhan Xiong and Wenyin Fu and Whitney Meers and Xavier Martinet and Xiaodong Wang and Xiaofang Wang and Xiaoqing Ellen Tan and Xide Xia and Xinfeng Xie and Xuchao Jia and Xuewei Wang and Yaelle Goldschlag and Yashesh Gaur and Yasmine Babaei and Yi Wen and Yiwen Song and Yuchen Zhang and Yue Li and Yuning Mao and Zacharie Delpierre Coudert and Zheng Yan and Zhengxing Chen and Zoe Papakipos and Aaditya Singh and Aayushi Srivastava and Abha Jain and Adam Kelsey and Adam Shajnfeld and Adithya Gangidi and Adolfo Victoria and Ahuva Goldstand and Ajay Menon and Ajay Sharma and Alex Boesenberg and Alexei Baevski and Allie Feinstein and Amanda Kallet and Amit Sangani and Amos Teo and Anam Yunus and Andrei Lupu and Andres Alvarado and Andrew Caples and Andrew Gu and Andrew Ho and Andrew Poulton and Andrew Ryan and Ankit Ramchandani and Annie Dong and Annie Franco and Anuj Goyal and Aparajita Saraf and Arkabandhu Chowdhury and Ashley Gabriel and Ashwin Bharambe and Assaf Eisenman and Azadeh Yazdan and Beau James and Ben Maurer and Benjamin Leonhardi and Bernie Huang and Beth Loyd and Beto De Paola and Bhargavi Paranjape and Bing Liu and Bo Wu and Boyu Ni and Braden Hancock and Bram Wasti and Brandon Spence and Brani Stojkovic and Brian Gamido and Britt Montalvo and Carl Parker and Carly Burton and Catalina Mejia and Ce Liu and Changhan Wang and Changkyu Kim and Chao Zhou and Chester Hu and Ching-Hsiang Chu and Chris Cai and Chris Tindal and Christoph Feichtenhofer and Cynthia Gao and Damon Civin and Dana Beaty and Daniel Kreymer and Daniel Li and David Adkins and David Xu and Davide Testuggine and Delia David and Devi Parikh and Diana Liskovich and Didem Foss and Dingkang Wang and Duc Le and Dustin Holland and Edward Dowling and Eissa Jamil and Elaine Montgomery and Eleonora Presani and Emily Hahn and Emily Wood and Eric-Tuan Le and Erik Brinkman and Esteban Arcaute and Evan Dunbar and Evan Smothers and Fei Sun and Felix Kreuk and Feng Tian and Filippos Kokkinos and Firat Ozgenel and Francesco Caggioni and Frank Kanayet and Frank Seide and Gabriela Medina Florez and Gabriella Schwarz and Gada Badeer and Georgia Swee and Gil Halpern and Grant Herman and Grigory Sizov and Guangyi and Zhang and Guna Lakshminarayanan and Hakan Inan and Hamid Shojanazeri and Han Zou and Hannah Wang and Hanwen Zha and Haroun Habeeb and Harrison Rudolph and Helen Suk and Henry Aspegren and Hunter Goldman and Hongyuan Zhan and Ibrahim Damlaj and Igor Molybog and Igor Tufanov and Ilias Leontiadis and Irina-Elena Veliche and Itai Gat and Jake Weissman and James Geboski and James Kohli and Janice Lam and Japhet Asher and Jean-Baptiste Gaya and Jeff Marcus and Jeff Tang and Jennifer Chan and Jenny Zhen and Jeremy Reizenstein and Jeremy Teboul and Jessica Zhong and Jian Jin and Jingyi Yang and Joe Cummings and Jon Carvill and Jon Shepard and Jonathan McPhie and Jonathan Torres and Josh Ginsburg and Junjie Wang and Kai Wu and Kam Hou U and Karan Saxena and Kartikay Khandelwal and Katayoun Zand and Kathy Matosich and Kaushik Veeraraghavan and Kelly Michelena and Keqian Li and Kiran Jagadeesh and Kun Huang and Kunal Chawla and Kyle Huang and Lailin Chen and Lakshya Garg and Lavender A and Leandro Silva and Lee Bell and Lei Zhang and Liangpeng Guo and Licheng Yu and Liron Moshkovich and Luca Wehrstedt and Madian Khabsa and Manav Avalani and Manish Bhatt and Martynas Mankus and Matan Hasson and Matthew Lennie and Matthias Reso and Maxim Groshev and Maxim Naumov and Maya Lathi and Meghan Keneally and Miao Liu and Michael L. Seltzer and Michal Valko and Michelle Restrepo and Mihir Patel and Mik Vyatskov and Mikayel Samvelyan and Mike Clark and Mike Macey and Mike Wang and Miquel Jubert Hermoso and Mo Metanat and Mohammad Rastegari and Munish Bansal and Nandhini Santhanam and Natascha Parks and Natasha White and Navyata Bawa and Nayan Singhal and Nick Egebo and Nicolas Usunier and Nikhil Mehta and Nikolay Pavlovich Laptev and Ning Dong and Norman Cheng and Oleg Chernoguz and Olivia Hart and Omkar Salpekar and Ozlem Kalinli and Parkin Kent and Parth Parekh and Paul Saab and Pavan Balaji and Pedro Rittner and Philip Bontrager and Pierre Roux and Piotr Dollar and Polina Zvyagina and Prashant Ratanchandani and Pritish Yuvraj and Qian Liang and Rachad Alao and Rachel Rodriguez and Rafi Ayub and Raghotham Murthy and Raghu Nayani and Rahul Mitra and Rangaprabhu Parthasarathy and Raymond Li and Rebekkah Hogan and Robin Battey and Rocky Wang and Russ Howes and Ruty Rinott and Sachin Mehta and Sachin Siby and Sai Jayesh Bondu and Samyak Datta and Sara Chugh and Sara Hunt and Sargun Dhillon and Sasha Sidorov and Satadru Pan and Saurabh Mahajan and Saurabh Verma and Seiji Yamamoto and Sharadh Ramaswamy and Shaun Lindsay and Shaun Lindsay and Sheng Feng and Shenghao Lin and Shengxin Cindy Zha and Shishir Patil and Shiva Shankar and Shuqiang Zhang and Shuqiang Zhang and Sinong Wang and Sneha Agarwal and Soji Sajuyigbe and Soumith Chintala and Stephanie Max and Stephen Chen and Steve Kehoe and Steve Satterfield and Sudarshan Govindaprasad and Sumit Gupta and Summer Deng and Sungmin Cho and Sunny Virk and Suraj Subramanian and Sy Choudhury and Sydney Goldman and Tal Remez and Tamar Glaser and Tamara Best and Thilo Koehler and Thomas Robinson and Tianhe Li and Tianjun Zhang and Tim Matthews and Timothy Chou and Tzook Shaked and Varun Vontimitta and Victoria Ajayi and Victoria Montanez and Vijai Mohan and Vinay Satish Kumar and Vishal Mangla and Vlad Ionescu and Vlad Poenaru and Vlad Tiberiu Mihailescu and Vladimir Ivanov and Wei Li and Wenchen Wang and Wenwen Jiang and Wes Bouaziz and Will Constable and Xiaocheng Tang and Xiaojian Wu and Xiaolan Wang and Xilun Wu and Xinbo Gao and Yaniv Kleinman and Yanjun Chen and Ye Hu and Ye Jia and Ye Qi and Yenda Li and Yilin Zhang and Ying Zhang and Yossi Adi and Youngjin Nam and Yu and Wang and Yu Zhao and Yuchen Hao and Yundi Qian and Yunlu Li and Yuzi He and Zach Rait and Zachary DeVito and Zef Rosnbrick and Zhaoduo Wen and Zhenyu Yang and Zhiwei Zhao and Zhiyu Ma},
      year={2024},
      eprint={2407.21783},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.21783}, 
}
@inproceedings{li2024eagle, 
	author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
	title = {{EAGLE}: Speculative Sampling Requires Rethinking Feature Uncertainty}, 
	booktitle = {International Conference on Machine Learning},
	year = {2024}
}
@inproceedings{li2024eagle2, 
	author = {Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang}, 
	title = {{EAGLE-2}: Faster Inference of Language Models with Dynamic Draft Trees}, 
	booktitle = {Empirical Methods in Natural Language Processing},
	year = {2024}
}
@misc{li2025eagle3scalinginferenceacceleration,
      title={{EAGLE-3}: Scaling up Inference Acceleration of Large Language Models via Training-Time Test}, 
      author={Yuhui Li and Fangyun Wei and Chao Zhang and Hongyang Zhang},
      year={2025},
      eprint={2503.01840},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01840}, 
}
@inproceedings{10.5555/3666122.3668142,
author = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
title = {Judging LLM-as-a-judge with MT-bench and Chatbot Arena},
year = {2023},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
abstract = {Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences. To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them. We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform. Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80\% agreement, the same level of agreement between humans. Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain. Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.},
booktitle = {Proceedings of the 37th International Conference on Neural Information Processing Systems},
articleno = {2020},
numpages = {29},
location = {New Orleans, LA, USA},
series = {NIPS '23}
}
@inproceedings{xia-etal-2024-unlocking,
    title = "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding",
    author = "Xia, Heming  and
      Yang, Zhe  and
      Dong, Qingxiu  and
      Wang, Peiyi  and
      Li, Yongqi  and
      Ge, Tao  and
      Liu, Tianyu  and
      Li, Wenjie  and
      Sui, Zhifang",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.456/",
    doi = "10.18653/v1/2024.findings-acl.456",
    pages = "7655--7671",
    abstract = "To mitigate the high inference latency stemming from autoregressive decoding in Large Language Models (LLMs), Speculative Decoding has emerged as a novel decoding paradigm for LLM inference. In each decoding step, this method first drafts several future tokens efficiently and then verifies them in parallel. Unlike autoregressive decoding, Speculative Decoding facilitates the simultaneous decoding of multiple tokens per step, thereby accelerating inference. This paper presents a comprehensive overview and analysis of this promising decoding paradigm. We begin by providing a formal definition and formulation of Speculative Decoding. Then, we organize in-depth discussions on its key facets, such as drafter selection and verification strategies. Furthermore, we present a comparative analysis of leading methods under third-party testing environments. We aim for this work to serve as a catalyst for further research on Speculative Decoding, ultimately contributing to more efficient LLM inference."
}
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
'''