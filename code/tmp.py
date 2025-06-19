from collections import defaultdict
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch
import pdb

# 1. 加载数据集
# dataset = load_dataset("json", data_files="/data/lvbb1/screen_qa-main/answers_and_bboxes/validation.json")


# ds 是一个 DatasetDict 对象，它包含一个名为 "val" 的分割（split），而 "val" 才是真正的 Dataset 对象（包含实际数据）。
ds = DatasetDict.from_json({'val': '/data/lvbb1/screen_qa-main/answers_and_bboxes/validation.json'})
ds = ds['val']
'''
DatasetDict({
    train: Dataset({
        features: ['image_id', 'image_width', 'image_height', 'question', 'ground_truth'],
        num_rows: 8614
    })
})
'''

key_counts = defaultdict(int)
for item in ds:
    # pdb.set_trace()
    ground_truth_list = item.get("ground_truth", None)
    if ground_truth_list:
        for gt_dict in ground_truth_list:  # 遍历每个字典
            for key in gt_dict.keys():  # 遍历字典的键
                key_counts[key] += 1

print("每个属性的出现次数:", dict(key_counts))

# 每个属性的出现次数: {'full_answer': 24051, 'ui_elements': 24051}， 注意的是, 每个ground_truth里面包含多个元素
# 'ui_elements'里面包括：'text', 'bounds', 'vh_index'
'''
{"image_id": 5, "image_width": 1080, "image_height": 1920, "question": "How many exercises in total are there to do?", 
"ground_truth": [{"full_answer": "There are 12 exercises in total to do.", 
                "ui_elements": [{"text": "12", "bounds": [509, 116, 569, 169], "vh_index": -1}]}, 
            {"full_answer": "There are total 12 exrecises to do.", 
            "ui_elements": [{"text": "12", "bounds": [503, 92, 583, 192], "vh_index": -1}]}, 
        {"full_answer": "There are 12 exercises in total to do.", 
        "ui_elements": [{"text": "12", "bounds": [499, 106, 577, 181], "vh_index": -1}]}]},
'''








{'image_id': 31, 'image_width': 1080, 'image_height': 1920, 
    'question': 'From whom are you protected?', 
    'ground_truth': 
        [{'full_answer': 'You are protected from unauthorized transactions.', 
            'ui_elements': [{'bounds': [424, 1078, 852, 1117], 
                    'text': 'unauthorized transactions', 'vh_index': -1}]}, 
        {'full_answer': 'You are protected from unauthorized transactions.', 
            'ui_elements': [{'bounds': [426, 1078, 852, 1113], 
                'text': 'unauthorized transactions', 'vh_index': -1}]}, 
        {'full_answer': 'You are protected from unauthorized transactions.', 
            'ui_elements': [{'bounds': [416, 1078, 854, 1115], 
                'text': 'unauthorized transactions', 'vh_index': -1}]}]}