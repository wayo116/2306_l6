from datalists import dlists 
import random

import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# import pandas as pd
import lightgbm as lgb

import csv
import os


def no_dataset(dlists, target_kaisu_lists):

    no_dataset = []
    for dlist_retu in range(6):

        target_kaisu_list = target_kaisu_lists[dlist_retu]
        
        for kaisu, dlist in enumerate(dlists):

            kaisu_limit = len(dlists)-max(target_kaisu_list)
            # print("kaisu_limit",kaisu_limit)
            if kaisu >= kaisu_limit:
                break

            tmp = []
            tmp.append(dlist[dlist_retu])
            for target_kaisu_youso in target_kaisu_list:
                tmp.append(dlists[kaisu+target_kaisu_youso][dlist_retu])

            if tmp != []:
                # no_dataset.append(tmp)

                list1 = tmp[1:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start=0.9*-1, range_end=0.9, yousosu=len(target_kaisu_list), listsu=100)
                for list2 in list2s:
                    # print("list2",list2)
                    result = [x + y for x, y in zip(list1, list2)]
                    # print("result",result)
                    result.insert(0, dlist[dlist_retu])
                    # print("result_in",result)
                    no_dataset.append(result)
         
    # print("no_dataset",no_dataset)
    print("no_dataset_len",len(no_dataset))

    return no_dataset


def no_dataset_test(dlists, target_kaisu_lists):

    no_dataset = []
    for dlist_retu in range(6):

        target_kaisu_list = target_kaisu_lists[dlist_retu]
        
        tmp = []
        for target_kaisu_youso in target_kaisu_list:
            tmp.append(dlists[target_kaisu_youso][dlist_retu])

        if tmp != []:
            # no_dataset.append(tmp)

            # list1 = tmp[1:]
            list1 = tmp[0:]
            # print("list1",list1)
            list2s = create_random_lists_float(range_start=0.9*-1, range_end=0.9, yousosu=len(target_kaisu_list), listsu=100)
            for list2 in list2s:
                # print("list2",list2)
                result = [x + y for x, y in zip(list1, list2)]
                # print("result",result)
                # result.insert(0, dlist[dlist_retu])
                # print("result_in",result)
                no_dataset.append(result)
         
    # print("no_dataset_test",no_dataset)
    print("no_dataset_test_len",len(no_dataset))

    return no_dataset


def create_random_lists(range_start, range_end, yousosu, listsu=6):

    random_lists = []
    for cnt in range(listsu):
        random.seed(cnt+1)
        random_list = random.sample(range(range_start, range_end + 1), yousosu)
        random_lists.append(random_list)

    # print("random_lists",random_lists)
    return random_lists

def create_random_lists_multi(range_start, range_end, yousosu, multisu, listsu=6):

    random_lists = []
    for dummy_cnt in range(listsu):
        tmp = []
        for cnt in range(multisu):
            random.seed(cnt+1)
            random_list = random.sample(range(range_start, range_end + 1), yousosu)
            tmp.append(random_list)
            print(tmp)
        random_lists.append(tmp)
    # print("random_lists",random_lists)
    return random_lists

def create_random_lists_float(range_start, range_end, yousosu, listsu=6):
    
    random_lists = []
    for cnt in range(listsu):
        random.seed(cnt+1)
        random_list = [round(random.uniform(range_start, range_end), 2) for _ in range(yousosu)]
        random_lists.append(random_list)

    # print("random_lists",random_lists)
    return random_lists


def light_gbm(data, data2):

    # 一列目のラベルを取得
    labels = [row[0] for row in data]

    # ラベルの個数をカウント
    label_counts = Counter(labels)

    # 一番個数が少ないラベルの数を取得
    min_label_count = min(label_counts.values())

    # 一番少ないラベルに合わせて他のラベルをフィルタリング
    balanced_data = []
    for label in label_counts.keys():
        filtered_data = [row for row in data if row[0] == label][:min_label_count]
        balanced_data.extend(filtered_data)

    # print("\nBalanced Data:")
    # for row in balanced_data:
    #     print(row)

    data = np.array(balanced_data)
    X = data[:,1:]
    print(X)
    y = data[:,0]
    print(y)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデル
    # max_depth=-1は無制限を意味する
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, objective='multiclass')
    model.fit(X_train, y_train)

    # 評価
    score = model.score(X_test, y_test)
    print("score", score)

    # 推論
    predictions = sorted(list(map(int, set(model.predict(data2)))))
    print("Predictions:", predictions)

    # 結果表示
    # for true_label, prediction in zip(y_test, predictions):
    #     print("True Label:", true_label)
    #     print("Prediction:", prediction)
    #     print()

    return score ,predictions
'''
csv_dir = "./result.csv"

# ファイルが存在する場合のみ削除
if os.path.exists(csv_dir):
    os.remove(csv_dir)
    print("ファイルを削除しました。")
else:
    print("指定されたファイルは存在しません。")

csv_dir = "./result.csv"
with open(csv_dir, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["st_cnt", "yousosu_cnt", "icchi", "dlists0", "score ,", "predictions","per"])

for st_cnt in range(1,10,1):
    for yousosu_cnt in range(4,14,1):
        range_start = 1
        range_end = 24
        yousosu = yousosu_cnt
        target_kaisu_lists = create_random_lists(range_start, range_end, yousosu)

        st = st_cnt
        dlists0 = dlists[st-1]

        dlists1 = dlists[st+1:500+st+1]
        data = no_dataset(dlists1, target_kaisu_lists)

        dlists2 = dlists[st:500+st]
        data2 = no_dataset_test(dlists2, target_kaisu_lists)

        score ,predictions = light_gbm(data, data2)

        icchi =  len(set(dlists0) & set(predictions))
        print(icchi)

        per = icchi/len(set(predictions))*100
        print(per)

        # csv_dir = "./result.csv"
        with open(csv_dir, "a", newline="") as file:
            writer = csv.writer(file)
            # writer.writerow(["yousosu_cnt", "icchi", "dlists0", "score ,", "predictions"])
            writer.writerow([st_cnt, yousosu_cnt, icchi, dlists0, score ,predictions, per])
'''
        



