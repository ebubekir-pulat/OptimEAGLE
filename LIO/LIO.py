print("\n\n*******************************\nStarting LIO.py\n\n")

from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch

base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

EAGLE_model_path = "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B"

# Preparing for LLM Instructed Optimisation (LIO) with EAGLE-3 Model
def EAGLE_LIO_init():
    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )

    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model


def suggestionsProcessor(LLM_suggestions):
    settings = ['"Tempearture":', '"Depth":', '"Top_k":', '"Top_k":', '"Threshold":', '"Top_p":', '"Total_token":']
    setting_values = []

    for setting in settings:
        index = LLM_suggestions.find(setting)

        while LLM_suggestions[index] != " ":
            index += 1

        index += 1

        setting_value = ""
        while LLM_suggestions[index] != " " and LLM_suggestions[index] != "\n":
            setting_value += LLM_suggestions[index]
            index += 1

        setting_values.append(float(setting_value))

    return setting_values


def template_getter():
    return "vicuna"

# LLM Instructed Optimisation (LIO) with EAGLE-3 Model for model setup
def EAGLE_LIO_model(model):
    prompt = "EAGLE-3 is a Speculative Decoding method, where the draft model attempts to emulate the target model (the LLM used), " \
    "to quickly generate draft tokens with high accuracy, that the target model will verify. EAGLE-3 uses self-drafting and a dynamic draft tree. Here " \
    "is some more background information: " \
    "To enable EAGLE’s effective utilisation of expanded training, EAGLE-3 abandons feature prediction and instead goes straight to " \
    "token prediction. Further, EAGLE-3 also incorporates the dynamic draft tree concept introduced in EAGLE-2. " \
    "EAGLE-3 enjoys a strong correlation between training data size, and resultant speed up as well as the mean number of " \
    "accepted draft tokens per forward pass, whereas EAGLE did not. Precisely, EAGLE-3 comprises of a training-time test " \
    "method which involves token prediction without feature prediction, allowing EAGLE-3 to utilise low, middle and high " \
    "features derived from the target model for token prediction, not being restricted to just top-layer features like EAGLE. " \
    "The authors argue this wider range of feature access supports the prediction of token_t+1 as well as token_t+2. " \
    "In action, EAGLE-3 extracts the low, middle and high-level features of the prior target model forward pass, then combine these " \
    "features into a vector which is processed by a fully connected layer that outputs a new heterogeneous and comprehensive feature, g. " \
    "A sequence of these special features is combined with the embedding of the last sampled token, and this is then processed " \
    "by a fully connected layer, then by a decoder to construct a vector a, which is sent to the drafter's LM head, " \
    "to ultimately produce the next token. For the next token, a sequence of g’s analogous to the current token sequence is " \
    "prepared to repeat the process, however, preferably the feature g of the recently sampled token would also form part of " \
    "this g sequence, but due to its unavailability, it is substituted by the previously formed a. " \
    "Provide one set of optimal settings for depth, top_k, threshold and total token for EAGLE-3, in JSON format, " \
    "to generally handle writing, roleplay, reasoning, math, coding, extraction, stem, humanities, translation, summarization, " \
    "qa, math_reasoning and rag tasks. Provide one set of settings to handle all tasks, not one for each task. " \
    "Be very clear in your decision and be concise.\n" \
    "Note: Depth = maximum draft length. \n" \
    "Top_k = maximum number of tokens drafted in each layer. \n" \
    "Total_token = number of draft tokens."

    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    conv = get_conversation_template(template_getter())
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    LIO_prompt = conv.get_prompt()
    input_ids = model.tokenizer([LIO_prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=1024)
    LIO_suggestions = model.tokenizer.decode(output_ids[0])
    print(LIO_suggestions)

# LLM Instructed Optimisation (LIO) with EAGLE-3 Model for eagenerate setup
def EAGLE_LIO_eagenerate(prompt_type, model):
    prompt = "EAGLE-3 is a Speculative Decoding method, where the draft model attempts to emulate the target model (the LLM used), " \
    "to quickly generate draft tokens with high accuracy, that the target model will verify. EAGLE-3 uses self-drafting and a dynamic draft tree. Here " \
    "is some more background information: " \
    "To enable EAGLE’s effective utilisation of expanded training, EAGLE-3 abandons feature prediction and instead goes straight to " \
    "token prediction. Further, EAGLE-3 also incorporates the dynamic draft tree concept introduced in EAGLE-2. " \
    "EAGLE-3 enjoys a strong correlation between training data size, and resultant speed up as well as the mean number of " \
    "accepted draft tokens per forward pass, whereas EAGLE did not. Precisely, EAGLE-3 comprises of a training-time test " \
    "method which involves token prediction without feature prediction, allowing EAGLE-3 to utilise low, middle and high " \
    "features derived from the target model for token prediction, not being restricted to just top-layer features like EAGLE. " \
    "The authors argue this wider range of feature access supports the prediction of token_t+1 as well as token_t+2. " \
    "In action, EAGLE-3 extracts the low, middle and high-level features of the prior target model forward pass, then combine these " \
    "features into a vector which is processed by a fully connected layer that outputs a new heterogeneous and comprehensive feature, g. " \
    "A sequence of these special features is combined with the embedding of the last sampled token, and this is then processed " \
    "by a fully connected layer, then by a decoder to construct a vector a, which is sent to the drafter's LM head, " \
    "to ultimately produce the next token. For the next token, a sequence of g’s analogous to the current token sequence is " \
    "prepared to repeat the process, however, preferably the feature g of the recently sampled token would also form part of " \
    "this g sequence, but due to its unavailability, it is substituted by the previously formed a. " \
    "Provide one set of optimal settings for temperature, top_p and top_k for EAGLE-3's generate function, in JSON format, " \
    f"for {prompt_type} tasks. Provide only one set of values. Provide a set of values no matter what and be concise.\n" \
    "Note: Top_k = maximum number of tokens drafted in each layer."

    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    conv = get_conversation_template(template_getter())
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    LIO_prompt = conv.get_prompt()
    input_ids = model.tokenizer([LIO_prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=512)
    LIO_suggestions = model.tokenizer.decode(output_ids[0])
    print(LIO_suggestions)


categories = ["writing",
              "roleplay",
              "reasoning",
              "math",
              "coding",
              "extraction",
              "stem",
              "humanities",
              "translation",
              "summarization",
              "qa",
              "math_reasoning",
              "rag",
              "long-context multi-doc QA",
              "long-context single-doc QA",
              "long-context summarization",
              "long-context few shot",
              "long-context synthetic",
              "long-context code completion"]

EAGLE_model = EAGLE_LIO_init()

# Getting LIO Suggestions with EAGLE-3 for model setup
print(f"\n\n********** MODEL SETTINGS **********")
EAGLE_LIO_model(EAGLE_model)

# Getting LIO Suggestions with EAGLE-3 for eagenerate setup
for i in range(13):
    print(f"\n\n********** {categories[i]} SETTINGS **********")
    EAGLE_LIO_eagenerate(categories[i], EAGLE_model)


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

4. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang,
J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” in Proceedings of
the 37th International Conference on Neural Information Processing Systems, ser. NIPS ’23. Red Hook, NY,
USA: Curran Associates Inc., 2023.

5. Y. Li, F. Wei, C. Zhang, and H. Zhang, “EAGLE-2: Faster inference of language models with dynamic
draft trees,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics,
Nov. 2024, pp. 7421–7432. [Online]. Available: https://aclanthology.org/2024.emnlp-main.422/

6. Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and
J. Li, “LongBench: A bilingual, multitask benchmark for long context understanding,” in Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku,
A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024,
pp. 3119–3137. [Online]. Available: https://aclanthology.org/2024.acl-long.172/

7. DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” 2025. [Online].
Available: https://arxiv.org/abs/2501.12948

'''

print("\n\n*******************************\nFinished Running LIO.py\n\n")