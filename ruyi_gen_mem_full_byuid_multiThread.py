import sys
from pymongo import MongoClient
from datetime import datetime
import os
from openai import OpenAI
import pandas as pd
import numpy as np
import random
import json
import openai
import re
import requests
import utils
import threading
import time


# prompt codes
prompt_ruyi_memory = "1744765940812"
prompt_memory_byday = "1744766106140"
prompt_summarize = "1744769200066"


# const
threadsNum = 20
bot = "如意"
scene = "chat"
appId_ruyi_test = "1457955735242624"   #如意测试环境appid
appId = "1457238095512704"  # 如意正式环境appid
# appId = "1457955735242624"   #如意测试环境appid


###############   需要修改参数：testFlag   ###############
# testFlag = 1 : 测试环境   0 ：正式环境
testFlag = True


# create multi threads
class getUserMemoryThreads(threading.Thread):
    def __init__(self, threadID, userCollections):
        super().__init__()
        self.threadID = threadID
        self.userCollections = userCollections

    def run(self):
        for num, user_id in enumerate(self.userCollections):
            try:
                user_id_str = str(user_id)
                user_document = utils.get_user_document_full(appId, user_id_str)

                df = pd.DataFrame(user_document)

                if df.empty:
                    print(f"{self.threadID}，{num}/{len(self.userCollections)}，用户{user_id}没有对话记录，即将跳过")
                    continue
                else:
                    print(f"{self.threadID}，{num}/{len(self.userCollections)}，已获取用户{user_id}的对话记录，{len(df)}条")
                
                grouped = df.sort_values(by='timestamp')
                grouped = df.groupby('userId')

                for userId, group in grouped:
                    user_json = utils.get_user_memory(userId, group, prompt_ruyi_memory, prompt_memory_byday, prompt_summarize, bot)

                    # No data for user or model failed to provide user memory in correct json format after retry
                    if user_json == None:
                        continue
                    else:
                        print(f"{self.threadID}，{num}/{len(self.userCollections)}，已生成 {userId} 的记忆json")
                    
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if testFlag == True:
                        user_memory_dto = utils.json_to_dto(user_json, current_time, appId_ruyi_test, scene)
                    else:
                        user_memory_dto = utils.json_to_dto(user_json, current_time, appId, scene)

                    if user_memory_dto == None:
                        continue
                    
                    print(f"{self.threadID}，{num}/{len(self.userCollections)}，已获得userId {userId} 的记忆dto，正在入库...")
                    if testFlag == True:
                        request = utils.call_lenovo_api(user_memory_dto)   #测试库
                    else:
                        request = utils.call_lenovo_api_write_formal(user_memory_dto)   # 正式库
                    #request = utils.call_lenovo_api(user_memory_dto)   #测试库
                    print(request)
                
            except Exception as e:
                print(f"{self.threadID}，错误：user_id {user_id} 发生了一个未知错误：{e}")


if __name__ == "__main__":
    ##############  需要计算全量记忆的用户列表  ######################### 
    user_collection1 = utils.get_users("/data/zhangch20/memory/ruyi_userids_20250612.xlsx")
     
    ##############  已经计算过全量记忆的用户uids ##########################
    # user_collection2 = utils.get_users("/data/zhangch20/memory/ruyi_fininised_uids_0526.xlsx")
    user_collection2 = utils.get_users("/data/zhangch20/memory/ruyi_2w_and_inner_test_userid.xlsx")

    user_set1=set(user_collection1)
    user_set2=set(user_collection2)
    user_collection = list(user_set1 - user_set2)

    print(f"提取到 {len(user_collection)} 个userId， 即将生成记忆\n")

    # user_collection = user_collection[:5]
    # user_collection = ['10279234045', '10304253987', '10279900222', '10294878578', '10253004787']
    # user_collection = ['10279900222']
    # print(user_collection[0])
    # os._exit(0)

    threadsAll = []
    start = 0
    length = round(len(user_collection) / threadsNum)
    for num in range(threadsNum):
        if num != threadsNum - 1:
            userCollections = user_collection[start: start + length]
        else:
            userCollections = user_collection[start:]

        threadsAll.append(getUserMemoryThreads("THREAD" + str(num), userCollections))

        start += length
    
    start = time.time()

    for eachThread in threadsAll:
        eachThread.start()

    for eachThread in threadsAll:
        eachThread.join()


    end = time.time()

    print("Over!!!!!!!!!!!!!")
    print("Cost time:{:.4f}s.".format(end - start))