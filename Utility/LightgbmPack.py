import os
import csv
import time

from datalists import dlists
from Utility.dataset import no_dataset_trainval_multi, no_dataset_test_multi, create_random_lists_multi, light_gbm, light_gbm_nogood, light_gbm_KFold, light_gbm_multi


class LightgbmPack():
    def __init__(self, makecsv=False):
        self.makecsv = makecsv

        if self.makecsv:
            self.csv_dir = 'result.csv'
            with open(self.csv_dir, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["kaisai",
                    "s_range_start",
                    "s_range_end",
                    "s_yousosu",
                    "s_multisu",
                    "s_randomkeisu",
                    "s_nmasi",
                    "t_range_start",
                    "t_range_end",
                    "t_yousosu",
                    "t_multisu",
                    "t_randomkeisu",
                    "t_nmasi",
                    "saisinkekka_list",
                    "predictions",
                    "l1l2_len",
                    "predictions_len",
                    "percent" 
                    ])

    def lightgbmpack(self, kaisai, saisinkekka_list, dlists, lgbm_model, **params):
        print('\n----lightGBMで予想----')
        start = time.time()

        #学習検証用
        range_start = params["train_params"]["range_start"]
        range_end = params["train_params"]["range_end"]
        yousosu = params["train_params"]["yousosu"]
        multisu = params["train_params"]["multisu"]
        randomkeisu = params["train_params"]["randomkeisu"]
        target_kaisu_lists = create_random_lists_multi(range_start, range_end, yousosu, multisu, randomkeisu)

        train_val_dlists = dlists[1:]
        nmasi = params["train_params"]["nmasi"]
        train_data = no_dataset_trainval_multi(train_val_dlists, target_kaisu_lists, nmasi)

        #テスト用
        range_start = params["test_params"]["range_start"]
        range_end = params["test_params"]["range_end"]
        yousosu = params["test_params"]["yousosu"]
        multisu = params["test_params"]["multisu"]
        randomkeisu = params["test_params"]["randomkeisu"]
        target_kaisu_lists = create_random_lists_multi(range_start, range_end, yousosu, multisu, randomkeisu)

        test_dlists = dlists[0:]
        nmasi = params["test_params"]["nmasi"]
        test_data = no_dataset_test_multi(test_dlists, target_kaisu_lists, nmasi)
        print("len(test_data)",len(test_data))

        #lightgbmで推論
        if lgbm_model == "light_gbm":
            score ,predictions = light_gbm(train_data, test_data)
            
        if lgbm_model == "light_gbm_nogood":
            score ,predictions = light_gbm_nogood(train_data, test_data)

        if lgbm_model == "light_gbm_KFold":
            score ,predictions = light_gbm_KFold(train_data, test_data)
        
        if lgbm_model == "light_gbm_multi":
            score ,predictions = light_gbm_multi(train_data, test_data)


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

        if self.makecsv:
            with open(self.csv_dir, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([kaisai,
                    params["train_params"]["range_start"],
                    params["train_params"]["range_end"],
                    params["train_params"]["yousosu"],
                    params["train_params"]["multisu"],
                    params["train_params"]["randomkeisu"],
                    params["train_params"]["nmasi"],
                    params["test_params"]["range_start"],
                    params["test_params"]["range_end"],
                    params["test_params"]["yousosu"],
                    params["test_params"]["multisu"],
                    params["test_params"]["randomkeisu"],
                    params["test_params"]["nmasi"],
                    saisinkekka_list,
                    predictions,
                    l1l2_len,
                    predictions_len,
                    percent,
                    ])

        print("処理時間",time.time() - start)

        return predictions
