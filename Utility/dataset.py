from datalists import dlists 
import random


# def no_dataset(dlists, no, target_kaisu_list):
#     no_dataset = []
#     for kaisu, dlist in enumerate(dlists):

#         kaisu_limit = len(dlists)-max(target_kaisu_list)
#         # print("kaisu_limit",kaisu_limit)
#         if kaisu >= kaisu_limit:
#             break

#         tmp = []
#         for dlist_idx, dlist_youso in enumerate(dlist):
#             if dlist_youso == no:
#                 tmp.append(no)
#                 for target_kaisu_youso in target_kaisu_list:
#                     tmp.append(dlists[kaisu+target_kaisu_youso][dlist_idx])

#         if tmp != []:
#             no_dataset.append(tmp)
         
#     print("no_dataset",no_dataset)
#     print("no_dataset_len",len(no_dataset))


def no_dataset(dlists, target_kaisu_lists):

    no_dataset = []
    for dlist_idx in range(6):

        target_kaisu_list = target_kaisu_lists[dlist_idx]
        
        for kaisu, dlist in enumerate(dlists):

            kaisu_limit = len(dlists)-max(target_kaisu_list)
            # print("kaisu_limit",kaisu_limit)
            if kaisu >= kaisu_limit:
                break

            tmp = []
            tmp.append(dlist[dlist_idx])
            for target_kaisu_youso in target_kaisu_list:
                tmp.append(dlists[kaisu+target_kaisu_youso][dlist_idx])

            if tmp != []:
                no_dataset.append(tmp)
         
    print("no_dataset",no_dataset)
    print("no_dataset_len",len(no_dataset))

    return no_dataset


def no_dataset_test(dlists, target_kaisu_lists):

    no_dataset = []
    for dlist_idx in range(6):

        target_kaisu_list = target_kaisu_lists[dlist_idx]
        
        tmp = []
        for target_kaisu_youso in target_kaisu_list:
            tmp.append(dlists[target_kaisu_youso][dlist_idx])

        if tmp != []:
            no_dataset.append(tmp)
         
    print("no_dataset_test",no_dataset)
    print("no_dataset_test_len",len(no_dataset))

    return no_dataset


def create_random_lists(range_start, range_end, yousosu, listsu=6):
    random_lists = []
    for cnt in range(listsu):
        random.seed(cnt+1)
        random_list = random.sample(range(range_start, range_end + 1), yousosu)
        random_lists.append(random_list)

    print("random_lists",random_lists)
    return random_lists


range_start = 1
range_end = 24
yousosu = 4
target_kaisu_lists = create_random_lists(range_start, range_end, yousosu)

dlists = dlists[1:500]
data = no_dataset(dlists, target_kaisu_lists)

dlists = dlists[0:500]
data2 = no_dataset_test(dlists, target_kaisu_lists)


import numpy as np
from collections import Counter

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

# 結果の表示
# print("Original Data:")
# for row in data:
#     print(row)

print("\nBalanced Data:")
for row in balanced_data:
    print(row)


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# import pandas as pd
import lightgbm as lgb

data = np.array(balanced_data)
X = data[:,1:]
print(X)
y = data[:,0]
print(y)

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル
# max_depth=-1は無制限を意味する
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=-1, objective='multiclass')
model.fit(X_train, y_train)

# 評価
print("score", model.score(X_test, y_test))

# 推論
predictions = model.predict(data2)
print("Predictions:", predictions)

# 結果表示
# for true_label, prediction in zip(y_test, predictions):
#     print("True Label:", true_label)
#     print("Prediction:", prediction)
#     print()


