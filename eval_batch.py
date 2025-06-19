from datasets import DatasetDict, load_dataset
import torch
from PIL import Image
import re
import json
import time
import yaml
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from metrics import sqa_s_metrics, sqa_uic_bb_metrics, sqa_uic_metrics, sqa_numeric_metrics
import model_adapters

MAX_RETRIES = 3
RETRY_DELAY = 1

def clean_llm_output(text):
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'#.*', '', text)
    return text.strip()

def build_prompt(question, image_description=None):
    """
    构建 prompt，支持传入图像描述或直接拼接文本
    """
    return f"""
You are a rigorous assistant for mobile app screen understanding. Given a question, you must output the answer in a strictly structured JSON format with the following rules:
### Output Format (Strictly Follow Below Structure)
Your answer MUST be a dictionary with:
1.full_answer: A concise text answer (natural language description), if your answer contains double quotes, you must escape them as \".
2.ui_elements: A list of UI elements, each containing:
    - text: Text displayed on the UI (e.g., button label).
    - bounds: Bounding box coordinates [y_min, x_min, y_max, x_max] (in pixels). Must contain exactly 4 numbers.
    - vh_index: If the UI element has a visible number label (e.g., "1", "2", "3"...), fill in the corresponding number. If not, set it to "-1".
Important:
- You MUST ensure double quotes or quotes inside strings are properly escaped as \".
- You MUST output a complete and valid JSON structure.

Example Output:
{{
    "full_answer": "The rating is 4.3 stars as shown on the screen.",
    "ui_elements": [
        {{
            "text": "4.3★ 1000K Downloads",
            "bounds": [424, 1078, 852, 1117],
            "vh_index": -1
        }}
    ]
}}

Now, given the question: "{question}", please provide your answer in the specified JSON format.
"""


def evaluate_batch(
    model_adapter,
    prompt_template,
    dataset,
    batch_size=8,
    max_workers=4
):
    data_size = len(dataset)
    all_results = []

    # 支持 batch 处理的 adapter 必须实现 batch_generate 方法
    for i in tqdm(range(0, data_size, batch_size), desc="Batch Processing"):
        batch_samples = dataset[i:i + batch_size]
        prompts = []
        images = []
        ground_truths_list = []

        for sample in batch_samples:
            iid = sample['image_id']
            img_path = f"/data/lvbb1/rico/combined/{iid}.jpg"
            question = sample['question']
            cur_prompt = prompt_template.format(question)
            prompts.append(cur_prompt)
            images.append(img_path)
            ground_truths_list.append(sample["ground_truth"])

        retry_count = 0
        success = False
        response = None

        while retry_count < MAX_RETRIES and not success:
            try:
                response = model_adapter.batch_generate(prompts, images)
                success = True
            except Exception as e:
                print(f"Batch inference failed (retry {retry_count + 1}/{MAX_RETRIES}): {e}")
                retry_count += 1
                time.sleep(RETRY_DELAY)

        if not success:
            response = ['{"full_answer":"<no answer>","ui_elements":[{"text":"<no answer>","bounds":[0,0,0,0],"vh_index":-1}]}' for _ in range(len(prompts))]

        # 解析结果
        for j, res_str in enumerate(response):
            res_str = clean_llm_output(res_str)
            try:
                data = json.loads(res_str)
            except json.JSONDecodeError:
                data = {"full_answer": "<no answer>", "ui_elements": [{"text": "<no answer>", "bounds": [0, 0, 0, 0], "vh_index": -1}]}

            best = {
                "full_answer": data.get("full_answer", "<no answer>"),
                "ui_elements": data.get("ui_elements", [{
                    "text": "<no answer>",
                    "bounds": [0, 0, 0, 0],
                    "vh_index": -1
                }])
            }

            # 提取指标
            ground_truths = ground_truths_list[j]
            full_answers = [gt['full_answer'] for gt in ground_truths]
            ui_texts = [[elem["text"] for elem in gt["ui_elements"]] for gt in ground_truths]
            ui_bb = [[(tuple(elem["bounds"]), elem["text"]) for elem in gt["ui_elements"]] for gt in ground_truths]
            vh_index = [[elem["vh_index"] for elem in gt["ui_elements"]] for gt in ground_truths]

            pred_text = best["full_answer"]
            pred_ui_texts = best["ui_elements"][0]["text"]
            pred_ui_bb = [(best["ui_elements"][0]["bounds"], best["ui_elements"][0]["text"])]
            pred_vh_index = best["ui_elements"][0]["vh_index"]

            em_s, f1_s = sqa_s_metrics(pred_text, full_answers)
            em_uic, f1_uic = sqa_uic_metrics(pred_ui_texts, ui_texts)
            bbox_f1, em_iou, f1_iou = sqa_uic_bb_metrics(pred_ui_bb, ui_bb)
            em_vh_index, f1_vh_index = sqa_numeric_metrics(pred_vh_index, vh_index)

            all_results.append((em_s, f1_s, em_uic, f1_uic, bbox_f1, em_iou, f1_iou, em_vh_index, f1_vh_index))

    # 计算平均指标
    fl_em_s = sum(x[0] for x in all_results) / data_size
    fl_f1_s = sum(x[1] for x in all_results) / data_size
    fl_em_uic = sum(x[2] for x in all_results) / data_size
    fl_f1_uic = sum(x[3] for x in all_results) / data_size
    fl_bbox_f1 = sum(x[4] for x in all_results) / data_size
    fl_em_iou = sum(x[5] for x in all_results) / data_size
    fl_f1_iou = sum(x[6] for x in all_results) / data_size
    fl_em_vh_index = sum(x[7] for x in all_results) / data_size
    fl_f1_vh_index = sum(x[8] for x in all_results) / data_size

    scores = {
        "em_s": fl_em_s,
        "f1_s": fl_f1_s,
        "em_uic": fl_em_uic,
        "f1_uic": fl_f1_uic,
        "bbox_f1": fl_bbox_f1,
        "em_iou": fl_em_iou,
        "f1_iou": fl_f1_iou,
        "em_vh_index": fl_em_vh_index,
        "f1_vh_index": fl_f1_vh_index
    }
    return scores



def main(args):
    model_config = yaml.load(open(f"configs/{args.model_name}.yaml"), Loader=yaml.FullLoader)
    prompt = build_prompt("")  # 使用通用模板
    dataset = DatasetDict.from_json({'val': '/data/lvbb1/screen_qa-main/answers_and_bboxes/validation.json'})['val']
    model_adapter = get_model_adapter(model_config, args.gpus)
    scores = evaluate_batch(model_adapter, prompt, dataset, batch_size=8)
    score_str = ', '.join([f"{k}: {v:.2f}" for k, v in scores.items()])
    print(f"Model: {args.model_name}, Scores: {score_str}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='qwen_vl', type=str,
                        choices=[file.split(".")[0] for file in os.listdir("configs") if file.endswith(".yaml")])
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument("--gpus", default="0", type=str, help="A single GPU like 1 or multiple GPUs like 0,2")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_path = os.path.join(args.output_path, args.model_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print(args)
    main(args)