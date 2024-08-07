import os
import json
import argparse
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, default="")
    parser.add_argument(
        "--result-file",
        type=str,
        default="",
    )
    parser.add_argument("--dataset", type=str, default="ccbench_20231003")
    return parser.parse_args()


def eval(result_file, annotation_file):
    results = [json.loads(q) for q in open(os.path.expanduser(result_file), "r")]
    annotations = [
        json.loads(q) for q in open(os.path.expanduser(annotation_file), "r")
    ]

    type_counts = {}
    correct_counts = {}
    i = 0
    for result, annotation in zip(results, annotations):
        data_type = annotation["category"]
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        matches = re.findall(r'[A-D]', result["text"])
        if matches:
            result_text = matches[0]
            if result_text == annotation["answer"]:
                correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

            
    
    total_count = 0
    total_correct = 0
    for data_type in sorted(type_counts.keys()):
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        print(f"{data_type} accuracy: {accuracy:.2f}%")
        # print(correct_counts[data_type], type_counts[data_type])
        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    # print(total_correct, total_count)
    total_accuracy = total_correct / total_count * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")

    return results


if __name__ == "__main__":
    args = get_args()
    print("====================== Eval MMBench ======================")
    print(f"Evaluating {args.dataset} ...")
    results = eval(args.result_file, args.annotation_file)
