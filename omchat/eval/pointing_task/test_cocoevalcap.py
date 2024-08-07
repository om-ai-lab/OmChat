from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

def eval_coco(annFile, resFile):
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    METEOR = cocoEval.eval["METEOR"]
    CIDEr = cocoEval.eval["CIDEr"]
    total = METEOR + CIDEr
    score = {"METEOR": METEOR, "CIDEr": CIDEr, "total": total}

    return score


question_file = "/data3/ljj/proj/MLLM_evals/data/refcoco_pointingtask/annotations/finetune_refcocog_val_annotation.json"
answers_file = "/data3/ljj/proj/omchat/outputs/refcoco_pointingtask/finetune_refcocog_val_answer-n117-v1_draw.json"
score = eval_coco(question_file, answers_file)

print(score)