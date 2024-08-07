from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

question_file = "/data3/ljj/proj/MLLM_evals/data/flickr30k/flickr30k_karpathy_test_cn.json"
answers_file = "/data3/ljj/proj/omchat/outputs/flickr30k/results-n104-cn.json"
coco = COCO(question_file)
coco_result = coco.loadRes(answers_file)
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.evaluate()

print(coco_eval.eval.items())