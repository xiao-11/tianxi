import os
import re
import time
import json
import yaml
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import DatasetDict, load_dataset
from PIL import Image
from metrics import sqa_s_metrics, sqa_uic_bb_metrics, sqa_uic_metrics, sqa_numeric_metrics
import requests
import logging
import model_adapters
from openai import OpenAI
import threading


# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1

def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(base_url="http://127.0.0.1:7879/v1", api_key="empty")
    return thread_local.client

# ===== 线程局部变量放在这里 =====
thread_local = threading.local()


def clean_llm_output(text):
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'#.*', '', text)
    return text.strip()

def collate_fn(batch):
    image_ids = [item["image_id"] for item in batch]
    image_widths = [item["image_width"] for item in batch]
    image_heights = [item["image_height"] for item in batch]
    questions = [item["question"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    return {
        "image_ids": image_ids,
        "image_widths": image_widths,
        "image_heights": image_heights,
        "questions": questions,
        "ground_truths": ground_truths,
    }

class ModelAdapterClient:
    def __init__(self, model_type, service_url="http://127.0.0.1:7879/v1"):
        self.service_url = service_url
        self.model_type = model_type

    def __call__(self, prompt, img_path):
        payload = {
            "prompt": prompt,
            "image_path": img_path
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.service_url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"Model inference failed with status {response.status_code}: {response.text}")


def evaluate_single_sample(args):
    idx_, config, prompt, dataset, gpus = args
    sample = dataset[idx_]
    iid = sample['image_id']
    img = f"/data/lvbb1/rico/combined/{iid}.jpg"
    question = sample['question']
    cur_prompt = prompt.format(question)

    # 获取远程服务地址
    model_path = config.get('model_path')
    tokenizer_path = config.get('tokenizer_path', model_path)
    device = f"cuda:{gpus}"

    ground_truths = sample["ground_truth"]
    full_answers = [gt['full_answer'] for gt in ground_truths]
    ui_texts = [[elem["text"] for elem in gt["ui_elements"]] for gt in ground_truths]
    ui_bb = [[(tuple(elem["bounds"]), elem["text"]) for elem in gt["ui_elements"]] for gt in ground_truths]
    vh_index = [[elem["vh_index"] for elem in gt["ui_elements"]] for gt in ground_truths]

    retry_count = 0
    success = False
    best = {
        "full_answer": "<no answer>",
        "ui_elements": [{
            "text": "<no answer>",
            "bounds": [0, 0, 0, 0],
            "vh_index": -1
        }]
    }

    client = get_client()
    model_list = client.models.list()
    model_id = model_list.data[0].id
    model_adapter = getattr(model_adapters, config['model_adapter'])(
        client, model_id,
    )

    while retry_count < MAX_RETRIES and not success:
        try:
            # client = OpenAI(base_url="http://127.0.0.1:7879/v1", api_key="empty")
            # model_list = client.models.list()
            # model_id = model_list.data[0].id
            # model_adapter = getattr(model_adapters, config['model_adapter'])(
            #     client, model_id,
            # )
            response = model_adapter(cur_prompt, img)
            response = clean_llm_output(response)
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
                retry_count += 1
                time.sleep(RETRY_DELAY)
                continue

            success = True
            if data.get("full_answer"):
                best["full_answer"] = data["full_answer"]
            else:
                success = False

            if data.get("ui_elements") and len(data["ui_elements"]) > 0:
                x = data["ui_elements"][0]
                if x.get("text"):
                    best["ui_elements"][0]["text"] = x["text"]
                else:
                    success = False
                if x.get("bounds") and len(x["bounds"]) == 4:
                    best["ui_elements"][0]["bounds"] = x["bounds"]
                else:
                    success = False
                if x.get("vh_index") is not None:
                    best["ui_elements"][0]["vh_index"] = x["vh_index"]
                else:
                    success = False
            else:
                success = False

        except Exception as e:
            logger.error(f"未知错误 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
            retry_count += 1
            time.sleep(RETRY_DELAY)
            continue

    pred_text = best["full_answer"]
    pred_ui_texts = best["ui_elements"][0]["text"]
    pred_ui_bb = [(best["ui_elements"][0]["bounds"], best["ui_elements"][0]["text"])]
    pred_vh_index = best["ui_elements"][0]["vh_index"]

    em_s, f1_s = sqa_s_metrics(pred_text, full_answers)
    em_uic, f1_uic = sqa_uic_metrics(pred_ui_texts, ui_texts)
    bbox_f1, em_iou, f1_iou = sqa_uic_bb_metrics(pred_ui_bb, ui_bb)
    em_vh_index, f1_vh_index = sqa_numeric_metrics(pred_vh_index, vh_index)

    return em_s, f1_s, em_uic, f1_uic, bbox_f1, em_iou, f1_iou, em_vh_index, f1_vh_index


def evaluate(config, prompt, dataset, gpus, max_workers=10):
    print('=' * 80)
    print('Prompt: ', prompt)
    data_size = len(dataset)
    st = time.time()

    args_list = [(idx_, config, prompt, dataset, gpus) for idx_ in range(data_size)]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(evaluate_single_sample, args): args[0] for args in args_list}
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Evaluating"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logger.error(f"Sample generated an exception: {exc}")

    fl_em_s = fl_f1_s = fl_em_uic = fl_f1_uic = fl_bbox_f1 = fl_em_iou = fl_f1_iou = 0
    fl_em_vh_index = fl_f1_vh_index = 0

    for res in results:
        fl_em_s += res[0]
        fl_f1_s += res[1]
        fl_em_uic += res[2]
        fl_f1_uic += res[3]
        fl_bbox_f1 += res[4]
        fl_em_iou += res[5]
        fl_f1_iou += res[6]
        fl_em_vh_index += res[7]
        fl_f1_vh_index += res[8]

    scores = {
        "em_s": fl_em_s / data_size,
        "f1_s": fl_f1_s / data_size,
        "em_uic": fl_em_uic / data_size,
        "f1_uic": fl_f1_uic / data_size,
        "bbox_f1": fl_bbox_f1 / data_size,
        "em_iou": fl_em_iou / data_size,
        "f1_iou": fl_f1_iou / data_size,
        "em_vh_index": fl_em_vh_index / data_size,
        "f1_vh_index": fl_f1_vh_index / data_size
    }

    end = time.time()
    print("Evaluation time:{} seconds".format(end - st))
    return scores


def main(args):
    config_path = f"configs/{args.model_name}.yaml"
    model_config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    prompt = """
        You are a rigorous assistant for mobile app screen understanding. Given a question, you must output the answer in a strictly structured JSON format with the following rules:
        ### **Output Format (Strictly Follow Below Structure)**
        Your answer MUST be a dictionary with:
        1.full_answer: A concise text answer (natural language description), if your answer contains double quotes, you must escape them as \".
        2.ui_elements: A list of UI elements, each containing:
            - text: Text displayed on the UI (e.g., button label).
            - bounds: Bounding box coordinates [y_min, x_min, y_max, x_max] (in pixels). **Must contain exactly 4 numbers, in the order: [top, left, bottom, right].**
                ❌ Invalid examples: [50, 401] (missing numbers), [50, 401) (wrong bracket), [50, 401, 80] (incomplete).**
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
        Now, given the question: "{}", please provide your answer in the specified JSON format.
        """

    dataset = DatasetDict.from_json({'val': '/data/lvbb1/screen_qa-main/answers_and_bboxes/validation.json'})['val']

    scores = evaluate(
        config=model_config,
        prompt=prompt,
        dataset=dataset,
        gpus=args.gpus,
        max_workers=10
    )

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
    # print(f"args.model_name:{args.model_name}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # print(args)
    main(args)