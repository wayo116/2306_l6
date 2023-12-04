# -*- coding: utf-8 -*-
"""main_v6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1POOwXUJfCtoLLXsqCVeTOaOcoutumX8Y
"""

!git clone https://wayo116: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@github.com/wayo116/2306_l6.git
#!pip install umap-learn
import os
import csv
import time

os.chdir('/content/2306_l6')

from datalists import dlists
from Utility.inner_outer import combi
from Dell6_v2 import Dell6
from Utility.LightgbmPack import LightgbmPack

start = time.time()

kaisai = 5
if kaisai == -1:
    #本番
    #最新結果がgitjubに登録済の時
    saisinkekka_list=[99,99,99,99,99,99]
    dlists = dlists
elif kaisai == 0:
    #最新結果がcolabにはあるが、gitjubには未登録の時
    saisinkekka_list=[11,15,17,33,36,41]
    dlists = dlists
elif kaisai > 0:
    saisinkekka_list = dlists[kaisai-1]
    dlists = dlists[kaisai:]
print("saisinkekka_list",saisinkekka_list)
print("dlists",dlists[:5])

bunkatu=5

print('\n----vol 1----')
dlists_end = 350
predictions_all = []
predictions_delall = []
lgbm_obj = LightgbmPack()

params = {"dataset_params":{"study_range_start":0,
                            "study_range_end":0.1,
                            "study_nmasi":10,
                            "test_range_start":-3,
                            "test_range_end":3,
                            "test_nmasi":10,
                            "bunseki_hani":6,
                            "flat_hani":25,
                            "test_dlists_hani":[0,1]},
             "lgbm_params":{"lgbm_model":"light_gbm_v2",
                            'num_leaves':4,
                            'learning_rate':0.05,
                            "n_estimators":100,
                            "max_depth":3,
                            "random_seed":777,
                            "cv":3,}}

predictions = lgbm_obj.lightgbmpack(kaisai, saisinkekka_list, dlists, dlists_end, **params)
predictions_all.extend(predictions)

print("saisinkekka_list",saisinkekka_list)
predictions_all = sorted(list(map(int, set(predictions_all))))
print("predictions_all_set",predictions_all)

l1 = saisinkekka_list
l2 = predictions_all
l1_l2_and = set(l1) & set(l2)
l1l2_len = len(l1_l2_and)
predictions_len = len(predictions_all)

if l1l2_len > 0 and predictions_len > 0:
    percent = round(l1l2_len/predictions_len*100)
    print(f"{l1l2_len}/{predictions_len}")
else:
    percent = 0
print("percent",percent)
print("\n")


delall = False
if delall == True:
    print('\n----vol 2----')
    dlists_end = 350
    predictions_delall = []
    lgbm_obj = LightgbmPack()

    params = {"dataset_params":{"study_range_start":0,
                                "study_range_end":0.1,
                                "study_nmasi":10,
                                "test_range_start":-0.1,
                                "test_range_end":0.1,
                                "test_nmasi":10,
                                "bunseki_hani":12,
                                "flat_hani":25,
                                "test_dlists_hani":[0,1]},
                "lgbm_params":{"lgbm_model":"light_gbm_v2",
                                'num_leaves':4,
                                'learning_rate':0.05,
                                "n_estimators":100,
                                "max_depth":3,
                                "random_seed":777,
                                "cv":3,}}

    predictions = lgbm_obj.lightgbmpack(kaisai, saisinkekka_list, dlists, dlists_end, **params)
    predictions_delall.extend(predictions)

    print("saisinkekka_list",saisinkekka_list)
    predictions_delall = sorted(list(map(int, set(predictions_delall))))
    print("predictions_delall_set",predictions_delall)

    l1 = saisinkekka_list
    l2 = predictions_delall
    l1_l2_and = set(l1) & set(l2)
    l1l2_len = len(l1_l2_and)
    predictions_len = len(predictions_delall)

    if l1l2_len > 0 and predictions_len > 0:
        percent = round(l1l2_len/predictions_len*100)
        print(f"{l1l2_len}/{predictions_len}")
    else:
        percent = 0
    print("percent",percent)
    print("\n")


print('\n----vol 1 2----')
predictions_unique = [item for item in predictions_all if item not in predictions_delall]
print("\npredictions_unique",predictions_unique)

l1 = saisinkekka_list
l2 = predictions_unique
l1_l2_and = set(l1) & set(l2)
l1l2_len = len(l1_l2_and)
predictions_len = len(predictions_unique)

if l1l2_len > 0 and predictions_len > 0:
    percent = round(l1l2_len/predictions_len*100)
    print(f"{l1l2_len}/{predictions_len}")
else:
    percent = 0
print("percent",percent)
print("\n")


pred_dlists = combi(predictions_unique,6)

#shori2は、pred_dlistsには組合せリストを入れる
outlist=Dell6(dlists, pred_dlists, saisinkekka_list, bunkatu).shori2()
#outlists.extend(outlist)
#print('outlist',outlist)

print("処理時間",time.time() - start)

from google.colab import drive
drive.mount('/content/drive')