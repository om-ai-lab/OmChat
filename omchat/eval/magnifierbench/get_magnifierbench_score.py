import requests
import json, os, re
import argparse
from tqdm import tqdm
import time


def get_chat_response(
    promot,
    gpt_model="gpt-4-0613",
    temperature=0,
    max_tokens=256,
    n=1,
    patience=5,
    sleep_time=5,
):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Your task is to judge whether the model response is correct to answer the given question or not.",
        },
        {"role": "user", "content": promot},
    ]

    body = {"model": gpt_model, "messages": messages}

    while patience > 0:
        patience -= 1
        try:
            response = requests.post(
                url="http://76.147.164.80:8080/chatgpt/v1/serving/chatcompletion",
                # "https://api.openai.com/v1/chat/completions",
                # headers=headers,
                # data=json.dumps(payload),
                json=body,
                timeout=30,
            )
            response.raise_for_status()
            response_data = response.json()

            prediction = response_data["choices"][0]["message"]["content"].strip()
            if prediction != "" and prediction is not None:
                return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)
            time.sleep(sleep_time)

    return ""


def prepare_query(model_answer_item, gpt_model):
    freeform_question = model_answer_item["freeform_question"]
    freeform_response = model_answer_item["freeform_response"]
    correct_answer = model_answer_item["freeform_answer"]

    # Formulating the prompt for ChatGPT
    prompt = f"Question: {freeform_question}\nModel Response: {freeform_response}\nGround Truth: {correct_answer}\nWill the model response be considered correct? You should only answer yes or no."

    # Querying ChatGPT
    chat_response = get_chat_response(prompt, gpt_model=gpt_model)

    return chat_response


def main(args):
    pred_ff = [
        json.loads(q) for q in open(os.path.expanduser(args.answers_ff), "r")
    ]  ## "freeform_answering"
    pred_mc = [
        json.loads(q) for q in open(os.path.expanduser(args.answers_mc), "r")
    ]  ## "multiple_choice"

    model_answer_path = args.answers_ff.replace("freeform_answering", "model_answer")
    result_path = args.answers_ff.replace("freeform_answering", "score")
    model_score_dict = {}
    model_answer = {}
    score = 0
    ff_score = 0
    if args.is_cn:
        ff_prompt = "\n用一个单词或短语回答问题。"
        mc_prompt = "\n直接使用给定选项的字母进行回答。"
    else:
        ff_prompt = "\nAnswer the question using a single word or phrase."
        mc_prompt = "\nAnswer with the option's letter from the given choices directly."

    for pff, pmc in zip(pred_ff, pred_mc):
        pred_id = pff["question_id"]
        pff_out = pff["text"]
        pmc_out = pmc["text"]
        question = pmc["prompt"]

        # count multiple_choice
        if pmc_out in args.options:
            ans_mc = pmc_out
        elif len(pmc_out) >= 3 and pmc_out[0] in args.options and pmc_out[1:3] == ". ":
            ans_mc = pmc_out[0]
        else:
            pattern = re.compile(r"The answer is ([A-Z]).")
            res = pattern.findall(pmc_out)
            if len(res) == 1:
                ans_mc = res[0]  # 'A', 'B', ...
            else:
                ans_mc = "FAILED"

        if ans_mc == pmc["answer"]:
            score += 1
        if args.is_cn:
            freeform_question = (
                (question.split("？")[0] + "？")
                .replace(ff_prompt, "")
                .replace("<image>\n", "")
                .strip()
            )
            options = question.split("？")[1].replace(mc_prompt, "")

        else:
            freeform_question = (
                (question.split("?")[0] + "?")
                .replace(ff_prompt, "")
                .replace("<image>\n", "")
                .strip()
            )
            options = question.split("?")[1].replace(mc_prompt, "")

        model_answer[pred_id] = {
            "question": pmc["prompt"].split("\n")[1],
            "options": options,
            "parsed_output": ans_mc,
            "answer": pmc["answer"],
            "freeform_question": freeform_question,
            "freeform_response": pff_out,
            "freeform_answer": pff["answer"],
        }
    with open(model_answer_path, "w") as f:
        json.dump(model_answer, f, indent=2, ensure_ascii=False)
    if args.is_cn:     
        print("====================== Eval magnifierbench (cn version)======================")
    else:
        print("====================== Eval magnifierbench (en version)======================")
    
    model_score_dict["score"] = score
    model_score_dict["total"] = len(pred_mc)
    model_score_dict["accuracy"] = score / len(pred_mc)
    print("magnifierbench accuracy: {}".format(model_score_dict["accuracy"]))

    if args.use_gpt:
        print(f"Start query {args.gpt_model} for free-form evaluation...")

        for data_id in tqdm(model_answer.keys(), desc=f"Querying {args.gpt_model}"):
            model_answer_item = model_answer[data_id]
            gpt_response = prepare_query(model_answer_item, args.gpt_model)
            if gpt_response.lower() == "yes":
                ff_score += 1
            elif gpt_response.lower() == "no":
                ff_score += 0
            else:
                print(f"Warning: {data_id} has invalid GPT-4 response: {gpt_response}")
                print(f"Skipping {data_id}")
                continue

        model_score_dict["freeform_score"] = ff_score
        model_score_dict["freeform_accuracy"] = ff_score / len(model_answer)
        print(
            "magnifierbench freeform_accuracy: {}".format(
                model_score_dict["freeform_accuracy"]
            )
        )

    with open(result_path, "w") as f:
        json.dump(model_score_dict, f, indent=2, ensure_ascii=False)

    print(f"Model answer saved to {model_answer_path}")
    print(f"Model score saved to {result_path}")
    print(json.dumps(model_score_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers-ff",
        type=str,
        default="/data/MLLM_evals/outputs/magnifierbench/results-llava1.5-freeform_answering.jsonl",
    )
    parser.add_argument(
        "--answers-mc",
        type=str,
        default="/data/MLLM_evals/outputs/magnifierbench/results-llava1.5-multiple_choice.jsonl",
    )
    parser.add_argument("--gpt-model", type=str, default="gpt-4-0613")
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument("--use-gpt", action="store_true")
    parser.add_argument("--is-cn", action="store_true")
    args = parser.parse_args()
    main(args)
