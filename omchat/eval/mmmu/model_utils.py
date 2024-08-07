from random import random
import re
from transformers import TextStreamer, PreTrainedTokenizer
import torch
from omchat.eval_scripts.make_context import make_context, make_context_yi
from PIL import Image
from omchat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
def extract_image_sequence(prompt):
    pattern = r"<image ([0-9]+)>"
    matches = re.findall(pattern, prompt)
    image_numbers = [int(match)-1 for match in matches]
    return image_numbers

def get_image_index(prompt):
    image_sequence = extract_image_sequence(prompt)
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
    #prompt = deal_with_prompt(prompt, False)
    #conv = conv_templates['vicuna_v1'].copy()
    #conv.append_message(conv.roles[0], prompt)
    #conv.append_message(conv.roles[1], None)
    #prompt = conv.get_prompt()
    system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    image = sample['image']
    #print (sample)
    #print (prompt)
    i_index = get_image_index(prompt)
    prompt = re.sub(r'<image \d+>', '<image>', prompt)
    #prompt = re.sub(r'<image \d+>', '', prompt)
    images = [i.half().cuda() for i in image]
    images = [images[index] for index in i_index] 


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
            max_new_tokens=4096,
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
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'
    print(response)
    return response.replace("<|im_end|>","")

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def llava_image_processor(raw_image, vis_processors=None):
    raw_image = [expand2square(i, tuple(int(x*255) for x in vis_processors.image_mean)) for i in raw_image]
    image_tensor = [vis_processors.preprocess(i, return_tensors='pt')['pixel_values'] for i in raw_image]
    return image_tensor
