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
        "percent",
        "dilist_end",
        "study_nmasi",
        "test_nmasi",
        "bunseki_hani",
        "test_dlists_hani_end",
        "num_leaves",
        "learning_rate",
        "n_estimators",
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

    for dilist_end in [50]:
        for study_nmasi in [50,70,90,130]:
            for test_nmasi in range(1,20,5):
                for bunseki_hani in range(4,28,4):
                    for test_dlists_hani_end in [1,2]:
                        for num_leaves in range(16,48,16):
                            for learning_rate in [0.01,0.05,0.1,0.5]:
                                for n_estimators in range(5,30,5):
                                    dlists_end = dilist_end
                                    bunkatu=5
                                    predictions_all = []
                                    predictions_delall = []
                                    lgbm_obj = LightgbmPack()

                                    print('\n----vol 1----')
                                    params = {"dataset_params":{"range_start": -0.1,
                                                                "range_end":0.1,
                                                                "study_nmasi":study_nmasi,
                                                                "test_nmasi":test_nmasi,
                                                                "bunseki_hani":bunseki_hani,
                                                                "test_dlists_hani_end":test_dlists_hani_end},

                                                "lgbm_params":{"lgbm_model": "light_gbm_multi",
                                                                'num_leaves': num_leaves,
                                                                'learning_rate': learning_rate,
                                                                "n_estimators":5,
                                                                "cv":3,}}

                                    predictions = lgbm_obj.lightgbmpack(kaisai, saisinkekka_list, dlists, dlists_end, **params)
                                    predictions_all.extend(predictions)

                                    print("saisinkekka_list",saisinkekka_list)
                                    predictions_all = sorted(list(map(int, set(predictions_all))))
                                    print("predictions_all_set",predictions_all)

                                    # dlists_end = 150
                                    # bunkatu=5
                                    # predictions_all = []
                                    # predictions_delall = []
                                    # lgbm_obj = LightgbmPack()

                                    # print('\n----vol 1----')
                                    # params = {"dataset_params":{"range_start": -0.1,
                                    #                             "range_end":0.1,
                                    #                             "study_nmasi":200,
                                    #                             "test_nmasi":20,
                                    #                             "bunseki_hani":4,
                                    #                             "test_dlists_hani_end":1},

                                    #             "lgbm_params":{"lgbm_model": "light_gbm_multi",
                                    #                             'num_leaves': 32,
                                    #                             'learning_rate': 0.01,
                                    #                             "n_estimators":5,
                                    #                             "cv":3,}}

                                    # predictions = lgbm_obj.lightgbmpack(kaisai, saisinkekka_list, dlists, dlists_end, **params)
                                    # predictions_all.extend(predictions)

                                    # print("saisinkekka_list",saisinkekka_list)
                                    # predictions_all = sorted(list(map(int, set(predictions_all))))
                                    # print("predictions_all_set",predictions_all)

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

                                    if l1l2_len > 0 and predictions_len > 0:
                                        percent = round(l1l2_len/predictions_len*100)
                                        print(f"{l1l2_len}/{predictions_len}")
                                    else:
                                        percent = 0
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
                                            dilist_end,
                                            study_nmasi,
                                            test_nmasi,
                                            bunseki_hani,
                                            test_dlists_hani_end,
                                            num_leaves,
                                            learning_rate,
                                            n_estimators,
                                            ])

                                    # pred_dlists = combi(predictions_all,6)

                                    # #shori2は、pred_dlistsには組合せリストを入れる
                                    # outlist=Dell6(dlists, pred_dlists, saisinkekka_list, bunkatu).shori2()
                                    # #outlists.extend(outlist)
                                    # #print('outlist',outlist)

print("処理時間",time.time() - start)