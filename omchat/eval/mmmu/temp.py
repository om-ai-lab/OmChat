import torch
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import json
from data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from eval_utils import parse_multi_choice_response, parse_open_response

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

print(1)
data_path = "/data3/qq/proj2/OmModelV4/omchat/MLLM_evals/data/MMMU"
split = "validation"
# run for each subject
sub_dataset_list = []

out_samples = dict()
for subject in CAT_SHORT2LONG.values():
    print(subject)
    samples = json.load(open("/data3/qq/proj2/OmModelV4/omchat/MLLM_evals/data/MMMU/" + subject + "/json/validation.json"))
    for sample in samples:
        out_samples[sample['id']] = sample["conversations"][-1]["value"]

print(len(out_samples))
#     sub_dataset = load_dataset(data_path, subject, split=split)
#     print(sub_dataset)
#     sub_dataset_list.append(sub_dataset)
#
# # merge all dataset
# dataset = concatenate_datasets(sub_dataset_list)
# print(dataset)

save_json("/data3/qq/proj2/OmModelV4/omchat/outputs/mmmu/test.json", out_samples)