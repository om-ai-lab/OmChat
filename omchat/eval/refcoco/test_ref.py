from torchvision.ops.boxes import box_area
import torch, re, json

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


outputs = json.load(open("/data3/ljj/proj/MLLM_evals/outputs/refcoco+/results-n104-refcoco+_val_normalized.json"))

PATTERN = re.compile(r'\((.*?)\),\((.*?)\)')
correct = total_cnt = 0
for i, output in enumerate(outputs):
    predict_bbox = re.findall(PATTERN, output['answer'])
    try:
        if ',' not in predict_bbox[0][0] or ',' not in predict_bbox[0][1]:
            predict_bbox = (0., 0., 0., 0.)
        else:
            x1, y1 = [
                float(tmp) for tmp in predict_bbox[0][0].split(',')
            ]
            x2, y2 = [
                float(tmp) for tmp in predict_bbox[0][1].split(',')
            ]
            predict_bbox = (x1, y1, x2, y2)
    except:
        predict_bbox = (0., 0., 0., 0.)
    target_bbox = torch.tensor(output['gt_bbox'],
                                dtype=torch.float32).view(-1, 4)
    predict_bbox = torch.tensor(predict_bbox,
                                dtype=torch.float32).view(-1, 4) 

    iou, _ = box_iou(predict_bbox, target_bbox)
    iou = iou.item()
    total_cnt += 1
    if iou >= 0.5:
        correct += 1


# print(f"Evaluating {args.dataset} ...")
print(f'Precision @ 1: {correct / total_cnt} \n')