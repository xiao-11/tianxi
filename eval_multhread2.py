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
import queue
from queue import Queue


# 日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
# RETRY_DELAY = 1

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



def evaluate(config, prompt, dataset, gpus, max_workers=10):
    print('=' * 80)
    print('Prompt: ', prompt)
    data_size = len(dataset)
    st = time.time()

    # 动态任务队列初始化
    task_queue = Queue()
    for idx in range(data_size):
        task_queue.put(idx)  # 将所有样本索引放入队列
    
    # 线程安全的锁（用于保护结果列表）
    result_lock = threading.Lock()
    results = [None] * data_size  # 按索引存储结果，保证顺序

    # 进度条
    pbar = tqdm(total=data_size, desc="Evaluating")

    # ---------------------- 定义线程工作函数 ----------------------
    def worker():
        client = get_client()
        model_list = client.models.list()
        model_id = model_list.data[0].id
        model_adapter = getattr(model_adapters, config['model_adapter'])(client, model_id,)

        while True:
            try:
                # 从队列中获取任务（block=True表示无任务时阻塞）
                idx = task_queue.get(block=True, timeout=1)
            except queue.Empty:
                # 队列空且超时，退出循环（避免永久阻塞）
                break

            try:
                # 处理单个样本（原evaluate_single_sample的核心逻辑）
                sample = dataset[idx]
                iid = sample['image_id']
                img_path = f"/data/lvbb1/rico/combined/{iid}.jpg"  # 可替换为动态路径
                question = sample['question']
                cur_prompt = prompt.format(question)

                # 模型推理逻辑（原代码核心）
                ground_truths = sample["ground_truth"]
                full_answers = [gt['full_answer'] for gt in ground_truths]
                ui_texts = [[elem["text"] for elem in gt["ui_elements"]] for gt in ground_truths]
                ui_bb = [[(tuple(elem["bounds"]), elem["text"]) for elem in gt["ui_elements"]] for gt in ground_truths]
                vh_index = [[elem["vh_index"] for elem in gt["ui_elements"]] for gt in ground_truths]

                retry_count = 0
                success = False
                best = {
                    "full_answer": "<no answer>",
                    "ui_elements": [{"text": "<no answer>", "bounds": [0, 0, 0, 0], "vh_index": -1}]
                }

                while retry_count < MAX_RETRIES and not success:
                    try:
                        response = model_adapter(cur_prompt, img_path)
                        response = clean_llm_output(response)
                        data = json.loads(response)
                        success = True

                        # 解析结果（原逻辑）
                        if data.get("full_answer"):
                            best["full_answer"] = data["full_answer"]
                        else:
                            success = False

                        if data.get("ui_elements") and len(data["ui_elements"]) > 0:
                            x = data["ui_elements"][0]
                            if x.get("text"):
                                best["ui_elements"][0]["text"] = x["text"]
                            if x.get("bounds") and len(x["bounds"]) == 4:
                                best["ui_elements"][0]["bounds"] = x["bounds"]
                            if x.get("vh_index") is not None:
                                best["ui_elements"][0]["vh_index"] = x["vh_index"]
                        else:
                            success = False

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析失败（尝试 {retry_count + 1}/{MAX_RETRIES}）: {e}")
                        retry_count += 1
                        # time.sleep(RETRY_DELAY)
                    except Exception as e:
                        logger.error(f"未知错误（尝试 {retry_count + 1}/{MAX_RETRIES}）: {e}")
                        retry_count += 1
                        # time.sleep(RETRY_DELAY)

                # 计算指标
                pred_text = best["full_answer"]
                pred_ui_texts = best["ui_elements"][0]["text"]
                pred_ui_bb = [(best["ui_elements"][0]["bounds"], best["ui_elements"][0]["text"])]
                pred_vh_index = best["ui_elements"][0]["vh_index"]

                em_s, f1_s = sqa_s_metrics(pred_text, full_answers)
                em_uic, f1_uic = sqa_uic_metrics(pred_ui_texts, ui_texts)
                bbox_f1, em_iou, f1_iou = sqa_uic_bb_metrics(pred_ui_bb, ui_bb)
                em_vh_index, f1_vh_index = sqa_numeric_metrics(pred_vh_index, vh_index)

                # 线程安全写入结果（按索引存储）
                with result_lock:
                    results[idx] = (em_s, f1_s, em_uic, f1_uic, bbox_f1, em_iou, f1_iou, em_vh_index, f1_vh_index)

            except Exception as e:
                logger.error(f"样本 {idx} 处理失败: {e}")
                with result_lock:
                    results[idx] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 失败样本标记为0
            finally:
                # 标记任务完成并更新进度条
                task_queue.task_done()
                pbar.update(1)


    # ---------------------- 启动线程并等待完成 ----------------------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 启动max_workers个线程，每个线程执行worker函数
        futures = [executor.submit(worker) for _ in range(max_workers)]
        # 等待所有线程完成（处理异常）
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"线程执行异常: {e}")

    pbar.close()  # 关闭进度条

    # ---------------------- 计算最终指标 ----------------------
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