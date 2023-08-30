import os
import csv
import time

from datalists import dlists
from Utility.dataset import no_dataset_multi, no_dataset_test_multi, create_random_lists_multi, light_gbm, lightgbm_cross, lightgbm_grid


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

    def lightgbmpack(self, kaisai, saisinkekka_list, dlists, **params):
        print('\n----lightGBMで予想----')
        start = time.time()

        #学習用
        range_start = params["train_params"]["range_start"]
        range_end = params["train_params"]["range_end"]
        yousosu = params["train_params"]["yousosu"]
        multisu = params["train_params"]["multisu"]
        randomkeisu = params["train_params"]["randomkeisu"]
        target_kaisu_lists = create_random_lists_multi(range_start, range_end, yousosu, multisu, randomkeisu)

        dlists1 = dlists[1:]
        nmasi = params["train_params"]["nmasi"]
        data = no_dataset_multi(dlists1, target_kaisu_lists, nmasi)

        #テスト用
        range_start = params["test_params"]["range_start"]
        range_end = params["test_params"]["range_end"]
        yousosu = params["test_params"]["yousosu"]
        multisu = params["test_params"]["multisu"]
        randomkeisu = params["test_params"]["randomkeisu"]
        target_kaisu_lists = create_random_lists_multi(range_start, range_end, yousosu, multisu, randomkeisu)

        dlists2 = dlists[0:]
        nmasi = params["test_params"]["nmasi"]
        data2 = no_dataset_test_multi(dlists2, target_kaisu_lists, nmasi)

        #lightgbmで推論
        score ,predictions = light_gbm(data, data2)
        # score ,predictions = lightgbm_cross(data, data2)
        # score ,predictions = lightgbm_grid(data, data2)

        l1 = saisinkekka_list
        l2 = predictions
        l1_l2_and = set(l1) & set(l2)
        l1l2_len = len(l1_l2_and)
        predictions_len = len(predictions)
        
        percent = round(l1l2_len/predictions_len*100)
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
