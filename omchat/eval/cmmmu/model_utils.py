from random import random
import re
from transformers import TextStreamer, PreTrainedTokenizer
import torch
from omchat.eval_scripts.make_context import make_context, make_context_yi
from omchat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
def extract_image_sequence(prompt, image_list):
    # pattern = r'<img="[^"]+">'
    # matches = re.findall(pattern, prompt)
    matches = re.findall(r'<img="([^"]+)"', prompt)
    # matches = set(matches)
    if matches == image_list:
        image_numbers = [i for i, match in enumerate(matches)]
    else:
        image_numbers = []
        for match in matches:
            position_list = [index for index, image in enumerate(image_list) if image == match]
            image_numbers.extend(position_list)
    return image_numbers

def get_image_index(prompt, image_list):
    image_sequence = extract_image_sequence(prompt, image_list)
    output = []
    for number in image_sequence:
        output.extend([number])  # Example logic: replicate each number twice in the output
    return output

def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None):
    from omchat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from omchat.conversation import conv_templates, SeparatorStyle

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    image_list = sample['image_list']
    valid_image_length = sum(1 for item in image_list if item is not None)
    valid_image_list = [item for item in image_list if item is not None]
    #prompt = deal_with_prompt(prompt, False)
    #conv = conv_templates['vicuna_v1'].copy()
    #conv.append_message(conv.roles[0], prompt)
    #conv.append_message(conv.roles[1], None)
    #prompt = conv.get_prompt()
    system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    image = sample['image']
    # print(len(image))
    # print (sample)
    # print (prompt)
    i_index = get_image_index(prompt, valid_image_list)
    # print(len(i_index))
    prompt = re.sub(r'<img="[^"]+">', '<image>', prompt)
    #prompt = re.sub(r'<image \d+>', '', prompt)
    images = [i.half().cuda() for i in image]
    
    # print(len(images))
    images = [images[index] for index in i_index] 
    # print(len(images))


    inp, context_tokens = make_context_yi(
                    tokenizer,
                    #DEFAULT_IMAGE_TOKEN*len(images) + "\n" + prompt.strip(),
                    prompt.strip(),
                    None,
                    system_message,
    )
    #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = torch.tensor([context_tokens]).cuda()
    keywords = ["<|im_end|>"]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    if image is not None:
        with torch.inference_mode():
            output_ids = model.generate(
            input_ids,
            #images=image.unsqueeze(0).half().cuda(),
            images=images,
            do_sample=True,
            temperature=1,
            streamer=streamer,
            max_new_tokens=1024,
            no_repeat_ngram_size=6,
            stopping_criteria=[stopping_criteria],
            use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        #response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        response = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
    else:  # multiple images actually
        if sample['question_type'] == '选择':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'
    print(response)
    return response.replace("<|im_end|>","")


def llava_image_processor(raw_image, vis_processors=None):
    image_tensor = [vis_processors.preprocess(i, return_tensors='pt')['pixel_values'] for i in raw_image]
    return image_tensor
