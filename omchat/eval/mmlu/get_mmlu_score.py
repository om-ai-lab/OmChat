import re
import argparse
import json
from tqdm.contrib import tzip
from tqdm import tqdm 
import os

TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]

def cal_mmlu(res):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("total cnt:", cnt, "\n")
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print("%s acc: %.2f " % (k, acc_sum_dict[k] / cnt_dict[k] * 100))
    print("AVERAGE acc: %.2f " % (acc_sum / cnt * 100))
    
def extract_choice(gen):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return None
    
    return res.group(1)


def eval_subject(anno_data, pred_data):
    score = []

    for pred, anno in tzip(pred_data, anno_data):
        response = pred["text"]
        
        pred_text = extract_choice(response)

        if pred_text == anno["conversations"][1]["value"]:
            correct = 1
        else:
            correct = 0
        score.append(correct)

    return score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", type=str)
    parser.add_argument("--result-path", type=str)
    parser.add_argument("--choose-eval-path", type=str, default="val")
    parser.add_argument("--model-id", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    dev_result = {}
    choose_eval = args.choose_eval_path
    for subject_name in tqdm(SUBJECTS):
        anno_data = json.load(open(os.path.join(args.annotation_path, choose_eval, f"{subject_name}_{choose_eval}.json")))
        pred_data = [json.loads(line) for line in open(os.path.join(args.result_path, choose_eval, args.model_id, f"{subject_name}_{choose_eval}_result.jsonl"))]
        score = eval_subject(anno_data, pred_data)
        dev_result[subject_name] = score
    print("====================== Eval MMLU ======================")
    cal_mmlu(dev_result)