from datasets import load_dataset, DatasetDict
import pdb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,)
from collections import defaultdict
from PIL import Image
from metrics import sqa_s_metrics, sqa_uic_bb_metrics, sqa_uic_metrics, sqa_numeric_metrics
from tqdm import tqdm
import model_adapters
import yaml
import argparse
import os
import json
import datasets
import re
import time
from multiprocessing import Pool


MAX_RETRIES = 3
RETRY_DELAY = 1


def clean_llm_output(text):
    # 移除 // 开头的注释
    text = re.sub(r'//.*', '', text)
    # 移除 # 开头的注释（如果需要）
    text = re.sub(r'#.*', '', text)
    return text.strip()


def collate_fn(batch):
    # 假设 batch 是一个列表，每个元素是一个样本（字典形式）
    image_ids = [item["image_id"] for item in batch]
    image_widths = torch.tensor([item["image_width"] for item in batch])
    image_heights = torch.tensor([item["image_height"] for item in batch])
    questions = [item["question"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]  # 假设是边界框列表

    # 如果需要将 question 转换为 tokenized 张量（需先定义 tokenizer）
    # questions_tensor = tokenizer(questions, padding=True, return_tensors="pt")

    # 返回一个字典，包含所有字段
    return {
        "image_ids": image_ids,
        "image_widths": image_widths,
        "image_heights": image_heights,
        "questions": questions,  # 或 questions_tensor（如果已分词）
        "ground_truths": ground_truths,
    }


def evaluate_single_sample(args):
    idx_, model_config, prompt, dataset, gpus = args

    model_path = model_config.get('model_path')
    tokenizer_path = model_config.get('tokenizer_path', model_path)
    device = f"cuda:{gpus}"
    model_name = model_path.split("/")[-1].lower()

    if "gpt" in model_name or "ui-tars" in model_name:
        from openai import OpenAI
        # client = OpenAI(api_key=os.getenv("empty"))
        client = OpenAI(base_url="http://127.0.0.1:7879/v1", api_key="empty")
        model_list = client.models.list()
        model_id = model_list.data[0].id
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(
            client, model_id,
        )
    elif "gemini" in model_name:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model_path)
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(model)
    elif "claude" in model_name:
        from anthropic import Anthropic

        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(
            client, model_path,
        )
    elif "llava" in model_name:
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import (
            get_model_name_from_path,
        )

        raw_model_name = get_model_name_from_path(model_path)
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, raw_model_name, device_map=None, device=device,
        )
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(
            model, tokenizer, context_len, image_processor, model_config['conv_mode']
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_adapter = getattr(model_adapters, model_config['model_adapter'])(model, tokenizer)
    

    sample = dataset[idx_]
    iid = sample['image_id']
    img = f"/data/lvbb1/rico/combined/{iid}.jpg"
    question = sample['question']
    cur_prompt = prompt.format(question)
    ground_truths = [gt for gt in sample["ground_truth"]]
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
    
    while retry_count < MAX_RETRIES and not success:
        try:
            response = model_adapter(cur_prompt, img)
            response = clean_llm_output(response)

            # print("原始响应")
            # print(response)
    
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
                retry_count += 1
                time.sleep(RETRY_DELAY)
                continue  # 跳过后续代码，直接重试

            success = True
            if data["full_answer"]:
                best["full_answer"] = data["full_answer"]
            else:
                successs = False
            if data["ui_elements"][0]:
                x = data["ui_elements"][0]
                if x["text"]:
                    best["ui_elements"][0]["text"] = x["text"]
                else:
                    success = False
                if x["bounds"] and len(x["bounds"]) == 4:
                    best["ui_elements"][0]["bounds"] = x["bounds"]
                else:
                    success = False
                if x["vh_index"]:
                    best["ui_elements"][0]["vh_index"] = x["vh_index"]
                else:
                    success = False
            else:
                success = False

        except (ValueError, KeyError) as e:
            print(f"数据字段缺失或格式错误 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
            retry_count += 1
            time.sleep(RETRY_DELAY)
            continue  # 重试
        except Exception as e:
            print(f"未知错误 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
            retry_count += 1
            time.sleep(RETRY_DELAY)
            continue  # 重试
    
    pred_text = best["full_answer"]
    pred_ui_texts = best["ui_elements"][0]["text"]
    pred_ui_bb = [(best["ui_elements"][0]["bounds"], best["ui_elements"][0]["text"])] 
    pred_vh_index = best["ui_elements"][0]["vh_index"]

    # 计算指标
    em_s, f1_s = sqa_s_metrics(pred_text, full_answers)
    em_uic, f1_uic = sqa_uic_metrics(pred_ui_texts, ui_texts)
    bbox_f1, em_iou, f1_iou = sqa_uic_bb_metrics(pred_ui_bb, ui_bb)
    em_vh_index, f1_vh_index = sqa_numeric_metrics(pred_vh_index, vh_index)

    return em_s, f1_s, em_uic, f1_uic, bbox_f1, em_iou, f1_iou, em_vh_index, f1_vh_index

    


def evaluate(
    model_config: dict,
    prompt: str,
    dataset: datasets.Dataset,
    gpus: str
):
    print('='*80)
    print('Prompt: ', prompt)
    data_size = len(dataset)

    st = time.time()
    args_list = [(idx_, model_config, prompt, dataset, gpus) for idx_ in range(data_size)]
    with Pool(processes=10) as pool:
        results = pool.map(evaluate_single_sample, args_list)

    fl_em_s, fl_f1_s, fl_em_uic, fl_f1_uic, fl_bbox_f1, fl_em_iou, fl_f1_iou = 0,0,0,0,0,0,0
    fl_em_vh_index, fl_f1_vh_index = 0, 0

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
    
    # 计算平均指标
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
    print("Evaluation time:{} seconds".format(end-st))
    return scores

    
def main(args):
    model_config = yaml.load(open(f"configs/{args.model_name}.yaml"), Loader=yaml.FullLoader)

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
        model_config=model_config,
        prompt=prompt,
        dataset=dataset,
        gpus=args.gpus
    )
    score_str = ', '.join([f"{k}: {v:.2f}" for k, v in scores.items()])
    print(f"Model: {args.model_name}, Scores: {score_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='qwen_vl',
        type=str,
        choices=[file.split(".")[0] for file in os.listdir("configs") if file.endswith(".yaml")],
    )
    parser.add_argument(
        '--output_path', 
        default='output', 
        type=str
    )
    parser.add_argument(
        "--gpus",
        default="0",
        type=str,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.output_path = os.path.join(args.output_path, args.model_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    main(args)

'''
好的输出例子：
{'full_answer': "The number of people who downloaded 'Legacy of Discord' is 1000K, as indicated by the download count displayed on the screen.", 'ui_elements': [{'text': '4.3★ 1000K Downloads', 'bounds': [424, 1078, 852, 1117], 'vh_index': -1}]}


坏的例子：
{'full_answer': "The rating of 'War Robots' is 4.6 stars, as shown next to the game's name in the list of apps on the screen.", 'ui_elements': [{'text': 'War Robots', 'bounds': [0, 0, 0, 0], 'vh_index': -1}]}
{'full_answer': "The application name is 'Lucky Block Mod for MCPE', as indicated by the yellow header at the top of the screen.", 'ui_elements': [{'text': 'Lucky Block Mod for MCPE', 'bounds': [0, 0, 0, 0], 'vh_index': -1}]}
'''