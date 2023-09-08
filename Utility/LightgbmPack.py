import os
import csv
import time

from datalists import dlists
from Utility.dataset import no_dataset_trainval_multi, no_dataset_test_multi, create_random_lists_multi, light_gbm, light_gbm_nogood, light_gbm_KFold, light_gbm_multi, light_gbm_v2


class LightgbmPack():
    def __init__(self):
        print("LightgbmPack")
        
    def lightgbmpack(self, kaisai, saisinkekka_list, dlists, dlists_end, **params):
        print('\n----lightGBMで予想----')
        start = time.time()

        dataset_params = params["dataset_params"]
        lgbm_params = params["lgbm_params"]

        #学習検証用
        train_val_dlists = dlists[1:dlists_end]
        train_data = no_dataset_trainval_multi(train_val_dlists, **dataset_params)
        print("len(train_data)",len(train_data))

        #テスト用
        test_data=[]
        for value in dlists[0]:
            cnt=0
            for row in train_data:
                if cnt < 3:
                    if row[0] == value:
                        test_data.append(row[1:])
                        cnt=cnt+1

        #テスト用
        #test_dlists = dlists[0:dlists_end]
        #test_data = no_dataset_test_multi(test_dlists, **dataset_params)
        print("len(test_data)",len(test_data))

        #lightgbmで推論
        if lgbm_params["lgbm_model"] == "light_gbm":
            score ,predictions = light_gbm(train_data, test_data, **lgbm_params)
            
        if lgbm_params["lgbm_model"] == "light_gbm_nogood":
            score ,predictions = light_gbm_nogood(train_data, test_data, **lgbm_params)

        if lgbm_params["lgbm_model"] == "light_gbm_KFold":
            score ,predictions = light_gbm_KFold(train_data, test_data, **lgbm_params)
        
        if lgbm_params["lgbm_model"] == "light_gbm_multi":
            score ,predictions = light_gbm_multi(train_data, test_data, **lgbm_params)

        if lgbm_params["lgbm_model"] == "light_gbm_v2":
            score ,predictions = light_gbm(train_data, test_data, **lgbm_params)


        # %計算
        l1 = saisinkekka_list
        l2 = predictions
        l1_l2_and = set(l1) & set(l2)
        l1l2_len = len(l1_l2_and)
        predictions_len = len(predictions)
        
        if l1l2_len > 0 and predictions_len > 0:
            percent = round(l1l2_len/predictions_len*100)
        else:
            percent = 0
        print("percent",percent)

        print("処理時間",time.time() - start)

        return predictions
