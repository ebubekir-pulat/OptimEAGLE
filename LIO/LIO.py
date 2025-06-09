from transformers import AutoTokenizer,  AutoModelForCausalLM
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch

base_model_paths = ["lmsys/vicuna-13b-v1.3",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.3-70B-Instruct"]

EAGLE_model_paths = ["yuhuili/EAGLE3-Vicuna1.3-13B",
                     "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
                     "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                     "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"]


# Preparing for LLM Instructed Optimisation (LIO)
def LIO_init(model_index):
    LLM_path = base_model_paths[model_index]
    # Below Code Block From: https://huggingface.co/learn/llm-course/chapter2/6?fw=pt, https://huggingface.co/docs/hub/transformers
    LLM_tokenizer = AutoTokenizer.from_pretrained(LLM_path)
    LLM = AutoModelForCausalLM.from_pretrained(LLM_path)
    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    LLM.eval()

    return LLM, LLM_tokenizer


# Preparing for LLM Instructed Optimisation (LIO) with EAGLE-3 Model
def EAGLE_LIO_init(model_index):
    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    model = EaModel.from_pretrained(
        base_model_path=base_model_paths[model_index],
        ea_model_path=EAGLE_model_paths[model_index],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1
    )

    # Below Code Line From: https://github.com/SafeAILab/EAGLE
    model.eval()
    return model


def suggestionsProcessor(LLM_suggestions):
    settings = ["Tempearture", "Depth", "Top_k", "Threshold"]
    setting_values = []

    for setting in settings:
        index = LLM_suggestions.find(setting)

        while LLM_suggestions[index] != " ":
            index += 1

        index += 1

        setting_value = ""
        while LLM_suggestions[index] != " ":
            setting_value += LLM_suggestions[index]

        setting_values.append(float(setting_value))

    return setting_values


# LLM Instructed Optimisation (LIO)
def LIO(prompt_type, LLM, LLM_tokenizer):
    LIO_prompt = f"For {prompt_type} tasks, provide the optimal settings for temperature, depth, top_k and threshold " \
                "for a draft model in an EAGLE-3 based Speculative Decoding method. The draft model attempts to emulate" \
                " the target model (the LLM used), to quickly generate draft tokens that the target model will verify. " \
                "EAGLE-3 uses self-drafting, and a dynamic draft tree, and prioritises speed of token generation. " \
                "Print the settings as: temperature: *value*, depth: *value* and so on, in an easily processed format."

    # Below Code Block From: https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM, https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM.forward.example
    inputs = LLM_tokenizer(LIO_prompt, return_tensors="pt")
    generate_ids = LLM.generate(inputs.input_ids, max_new_tokens=1024)
    LIO_suggestions_raw = LLM_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    LIO_suggestions = suggestionsProcessor(LIO_suggestions_raw)
    return LIO_suggestions


# LLM Instructed Optimisation (LIO) with EAGLE-3 Models
def EAGLE_LIO(prompt_type, model):
    prompt = f"For {prompt_type} tasks, provide the optimal settings for temperature, depth, top_k and threshold " \
                "for a draft model in an EAGLE-3 based Speculative Decoding method. The draft model attempts to emulate" \
                " the target model (the LLM used), to quickly generate draft tokens that the target model will verify. " \
                "EAGLE-3 uses self-drafting, and a dynamic draft tree, and prioritises speed of token generation. " \
                "Print the settings as: temperature: *value*, depth: *value* and so on, in an easily processed format."

    # Below Code Block From: https://github.com/SafeAILab/EAGLE
    conv = get_conversation_template(template_getter(model_index))
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    LIO_prompt = conv.get_prompt()
    input_ids = model.tokenizer([LIO_prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=1024)
    LIO_suggestions_raw = model.tokenizer.decode(output_ids[0])
    LIO_suggestions = suggestionsProcessor(LIO_suggestions_raw)
    return LIO_suggestions


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

model_index = 1

# Getting LIO Suggestions with EAGLE-3
EAGLE_model = EAGLE_LIO_init(model_index)
EAGLE_LIO_suggestions_set = {}

for i in range(len(categories)):
    # Getting EAGLE LIO Suggestions
    LIO_suggestions = LIO(categories[i], EAGLE_model)
    EAGLE_LIO_suggestions_set[categories[i]] = LIO_suggestions
    print(LIO_suggestions)

for category in EAGLE_LIO_suggestions_set:
    print(category, ": ", EAGLE_LIO_suggestions_set[category])

'''

# Getting LIO Suggestions without EAGLE3 Model
LLM, LLM_tokenizer = LIO_init(model_index)
LIO_suggestions_set = {}

for i in range(len(categories)):
    # Getting LIO Suggestions
    LIO_suggestions = LIO(categories[i], LLM, LLM_tokenizer)
    LIO_suggestions_set[categories[i]] = LIO_suggestions

    print(LIO_suggestions)

for category in LIO_suggestions_set:
    print(category, ": ", LIO_suggestions_set[category])

'''

''' 
References

1. 

2.

3.

4.

5.

6.

7.

8.

9.

10. 

'''