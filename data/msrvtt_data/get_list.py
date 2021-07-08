# @Time : 2021/4/26 3:15 下午 
# @Author : Han Fang
# @File: get_list.py 
# @Version: 2021/4/26 3:15 下午 get_list.py

import json
import os
import pandas as pd


train_list = []
for i in range(0, 6513):
    train_list.append('video' + str(i))
val_list = []
for i in range(6513, 7010):
    val_list.append('video' + str(i))
test_list = []
for i in range(7010,10000):
    test_list.append('video' + str(i))

trainframe = pd.DataFrame({'video_id':train_list})
trainframe.to_csv("MSRVTT_train.full.csv",index=False,sep=',')

valframe = pd.DataFrame({'video_id':val_list})
valframe.to_csv("MSRVTT_val.full.csv",index=False,sep=',')

testframe = pd.DataFrame({'video_id':val_list})
testframe.to_csv("MSRVTT_test.full.csv",index=False,sep=',')