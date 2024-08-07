"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import json
import yaml
import re

# DATA SAVING

def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall(r"<img='(.*?)'>", text)
    return matches
    
def process_single_sample_cmmmu(data):
    question = data['question']
    o_imgs_paths = []
    for option in [data['option1'], data['option2'], data['option3'], data['option4']]:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    images = []
    if data["image_1"]:
        images.append(data["image_1"].convert("RGB"))
    if data["image_2"]:
        images.append(data["image_2"].convert("RGB"))
    if data["image_3"]:
        images.append(data["image_3"].convert("RGB"))
    if data["image_4"]:
        images.append(data["image_4"].convert("RGB"))
    if data["image_5"]:
        images.append(data["image_5"].convert("RGB"))

    # print (len(images), len(o_imgs_paths))

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'option1': data['option1'], 'option2': data['option2'], 'option3': data['option3'], 'option4': data['option4'], 'answer': data['answer'],
             'image': None, 'question_type': data['type'], 'image_list': [data['image_1_filename'], data['image_2_filename'], data['image_3_filename'], data['image_4_filename'], data['image_5_filename']]}
    else:
        return {'id': data['id'], 'question': question, 'option1': data['option1'], 'option2': data['option2'], 'option3': data['option3'], 'option4': data['option4'], 'answer': data['answer'],
             'image': images, 'question_type': data['type'], 'image_list': [data['image_1_filename'], data['image_2_filename'], data['image_3_filename'], data['image_4_filename'], data['image_5_filename']]}     


# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)

def save_jsonl(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')
            
def save_args(args, path_dir):
    argsDict = args.__dict__
    with open(path_dir + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

# DATA PROCESSING
def construct_prompt_cmmmu(sample, config):
    question = sample['question']
    options = [sample['option1'], sample['option2'], sample['option3'], sample['option4']]
    example = ""

    if sample['question_type'] == '选择':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format'][0]
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][0].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        # print(sample["id"])
        # print(ord(sample['answer'].upper()))
        # res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
        gt_content = ""
        for option in sample['answer'].upper():
            gt_content += options[ord(option) - ord('A')]
        res_dict['gt_content'] = gt_content
        # print("+++++++++++++++", res_dict['gt_content'])
        
        
    elif sample['question_type'] == '判断':
        empty_prompt_sample_structure = config['T/F_example_format'][0]
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][1].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']
        # print("===============", res_dict['gt_content'])
        
    elif sample['question_type'] == '填空':
        empty_prompt_sample_structure = config['short_ans_example_format'][0]
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'][2].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']
        # print("!!!!!!!!!!!!", res_dict['gt_content'])

    res_dict.update(sample)
    return res_dict
