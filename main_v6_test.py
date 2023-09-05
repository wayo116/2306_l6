# !git clone https://wayo116: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@github.com/wayo116/2306_l6.git

import os
import csv
import time

# os.chdir('/content/2306_l6')

from datalists import dlists
from Utility.inner_outer import combi
from Dell6_v2 import Dell6
from Utility.LightgbmPack import LightgbmPack

start = time.time()

csv_dir = 'result.csv'
with open(csv_dir, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["kaisai",
        "saisinkekka_list",
        "predictions_unique",
        "l1l2_len",
        "predictions_len",
        "percent" 
        ])

for kaisai in range(1,6):
    # kaisai = -1
    if kaisai == -1:
        #本番
        #最新結果がgitjubに登録済の時
        saisinkekka_list=[99,99,99,99,99,99]
        dlists = dlists
    elif kaisai == 0:
        #最新結果がcolabにはあるが、gitjubには未登録の時
        saisinkekka_list=[9,11,12,14,27,38]
        dlists = dlists
    elif kaisai > 0:
        saisinkekka_list = dlists[kaisai-1]
        dlists = dlists[kaisai:]
    print("saisinkekka_list",saisinkekka_list)
    print("dlists",dlists[:5])

    dlists_end = 500
    bunkatu=5
    predictions_all = []
    predictions_delall = []
    lgbm_obj = LightgbmPack()

    print('\n----vol 1----')
    params = {"dataset_params":{"range_start": -0.1,
                                "range_end":0.1,
                                "study_nmasi":6,
                                "test_nmasi":1,
                                "bunseki_hani":5,
                                "test_dlists_hani_end":6},

                "lgbm_params":{"lgbm_model": "light_gbm_multi",
                                'num_leaves': 32,
                                'learning_rate': 0.1,
                                "n_estimators":10,
                                "cv":3,}}

    predictions = lgbm_obj.lightgbmpack(kaisai, saisinkekka_list, dlists, dlists_end, **params)
    predictions_all.extend(predictions)

    print("saisinkekka_list",saisinkekka_list)
    predictions_all = sorted(list(map(int, set(predictions_all))))
    print("predictions_all_set",predictions_all)

    '''
    print('\n----vol 1del----')
    params = {"dataset_params":{"range_start": -0.1,
                                "range_end":0.1,
                                "study_nmasi":6,
                                "test_nmasi":1,
                                "bunseki_hani":5,
                                "test_dlists_hani_end":6},

                "lgbm_params":{"lgbm_model": "light_gbm_multi",
                                'num_leaves': 32,
                                'learning_rate': 0.1,
                                "n_estimators":10,
                                "cv":3,}}

    predictions = lgbm_obj.lightgbmpack(kaisai, saisinkekka_list, dlists, dlists_end, **params)
    predictions_delall.extend(predictions)

    print("saisinkekka_list",saisinkekka_list)
    predictions_delall = sorted(list(map(int, set(predictions_delall))))
    print("predictions_delall_set",predictions_delall)
    '''

    predictions_unique = [item for item in predictions_all if item not in predictions_delall]
    print("predictions_unique",predictions_unique)

    l1 = saisinkekka_list
    l2 = predictions_unique
    l1_l2_and = set(l1) & set(l2)
    l1l2_len = len(l1_l2_and)
    predictions_len = len(predictions_unique)

    percent = round(l1l2_len/predictions_len*100)
    print(f'{l1l2_len}/{predictions_len}')
    print("percent",percent)
    print("\n")

    with open(csv_dir, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([kaisai,
            saisinkekka_list,
            predictions_unique,
            l1l2_len,
            predictions_len,
            percent,
            ])

    # pred_dlists = combi(predictions_all,6)

    # #shori2は、pred_dlistsには組合せリストを入れる
    # outlist=Dell6(dlists, pred_dlists, saisinkekka_list, bunkatu).shori2()
    # #outlists.extend(outlist)
    # #print('outlist',outlist)

print("処理時間",time.time() - start)