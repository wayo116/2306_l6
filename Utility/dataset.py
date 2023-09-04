from datalists import dlists 
import random

import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split, KFold, cross_val_score
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
import lightgbm as lgb

import csv
import os


def no_dataset_trainval_multi(dlists, target_kaisu_lists, nmasi):

    # no_dataset = []
    # for dlist_retu in range(6):

    #     #target_kaisu_list = target_kaisu_lists[dlist_retu]
    #     target_kaisu_lists_lists = target_kaisu_lists[dlist_retu]
        
    #     for target_kaisu_list in target_kaisu_lists_lists:
    #         for kaisu, dlist in enumerate(dlists):
    
    #             kaisu_limit = len(dlists)-max(target_kaisu_list)
    #             # print("kaisu_limit",kaisu_limit)
    #             if kaisu >= kaisu_limit:
    #                 break
    
    #             tmp = []
    #             tmp.append(dlist[dlist_retu])
    #             for target_kaisu_youso in target_kaisu_list:
    #                 tmp.append(dlists[kaisu+target_kaisu_youso][dlist_retu])

    #             if tmp != []:
    #                 # no_dataset.append(tmp)
    
    #                 list1 = tmp[1:]
    #                 # print("list1",list1)
    #                 list2s = create_random_lists_float(range_start=0.1*-1, range_end=0.1, yousosu=len(target_kaisu_list), listsu=nmasi)
    #                 # list2s = create_random_lists_float(range_start=0, range_end=0, yousosu=len(target_kaisu_list), listsu=nmasi)
    #                 for list2 in list2s:
    #                     # print("list2",list2)
    #                     result = [x + y for x, y in zip(list1, list2)]
    #                     # print("result",result)
    #                     result.insert(0, dlist[dlist_retu])
    #                     # print("result_in",result)
    #                     no_dataset.append(result)
         
    # # print("no_dataset",no_dataset)
    # print("no_dataset_len",len(no_dataset))


    no_dataset = []
    for kaisu, dlist in enumerate(dlists):

        for dlist_retu in range(6):

            kaisu_limit = len(dlists)-25
            # print("kaisu_limit",kaisu_limit)
            if kaisu >= kaisu_limit:
                break

            tmp = []
            tmp.append(dlist[dlist_retu])
            std3 = np.std(dlists[kaisu+1:kaisu+1+3, dlist_retu])
            std5 = np.std(dlists[kaisu+1:kaisu+1+5, dlist_retu])
            std10 = np.std(dlists[kaisu+1:kaisu+1+10, dlist_retu])
            std25 = np.std(dlists[kaisu+1:kaisu+1+25, dlist_retu])
            min25 = np.min(dlists[kaisu+1:kaisu+1+25, dlist_retu])
            tmp.append(std3)
            tmp.append(std5)
            tmp.append(std10)
            tmp.append(std25)
            tmp.append(min25)
            if tmp != []:
                # no_dataset.append(tmp)

                list1 = tmp[1:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start=0.1*-1, range_end=0.1, yousosu=len(tmp), listsu=nmasi)
                # list2s = create_random_lists_float(range_start=0, range_end=0, yousosu=len(target_kaisu_list), listsu=nmasi)
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


def no_dataset_test_multi(dlists, target_kaisu_lists, nmasi):

    # no_dataset = []
    # for dlist_retu in range(6):

    #     #target_kaisu_list = target_kaisu_lists[dlist_retu]
    #     target_kaisu_lists_lists = target_kaisu_lists[dlist_retu]

    #     for target_kaisu_list in target_kaisu_lists_lists:
    #         tmp = []
    #         for target_kaisu_youso in target_kaisu_list:
    #             tmp.append(dlists[target_kaisu_youso][dlist_retu])
    
    #         if tmp != []:
    #             # no_dataset.append(tmp)
    
    #             # list1 = tmp[1:]
    #             list1 = tmp[0:]
    #             # print("list1",list1)
    #             list2s = create_random_lists_float(range_start=0.1*-1, range_end=0.1, yousosu=len(target_kaisu_list), listsu=nmasi)
    #             # list2s = create_random_lists_float(range_start=0, range_end=0, yousosu=len(target_kaisu_list), listsu=nmasi)
    #             for list2 in list2s:
    #                 # print("list2",list2)
    #                 result = [x + y for x, y in zip(list1, list2)]
    #                 # print("result",result)
    #                 # result.insert(0, dlist[dlist_retu])
    #                 # print("result_in",result)
    #                 no_dataset.append(result)
         
    # # print("no_dataset_test",no_dataset)
    # print("no_dataset_test_len",len(no_dataset))

    no_dataset = []
    for kaisu, dlist in enumerate(dlists[0:1]):
        
        for dlist_retu in range(6):

            kaisu_limit = len(dlists)-25
            # print("kaisu_limit",kaisu_limit)
            if kaisu >= kaisu_limit:
                break

            tmp = []
            # tmp.append(dlist[dlist_retu])
            std3 = np.std(dlists[kaisu:kaisu+3, dlist_retu])
            std5 = np.std(dlists[kaisu:kaisu+5, dlist_retu])
            std10 = np.std(dlists[kaisu:kaisu+10, dlist_retu])
            std25 = np.std(dlists[kaisu:kaisu+25, dlist_retu])
            tmp.append(std3)
            tmp.append(std5)
            tmp.append(std10)
            tmp.append(std25)

            if tmp != []:
                # no_dataset.append(tmp)

                list1 = tmp[0:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start=0.1*-1, range_end=0.1, yousosu=len(tmp), listsu=nmasi)
                # list2s = create_random_lists_float(range_start=0, range_end=0, yousosu=len(target_kaisu_list), listsu=nmasi)
                for list2 in list2s:
                    # print("list2",list2)
                    result = [x + y for x, y in zip(list1, list2)]
                    # print("result",result)
                    # result.insert(0, dlist[dlist_retu])
                    # print("result_in",result)
                    no_dataset.append(result)

    # print("no_dataset",no_dataset)
    print("no_dataset_len",len(no_dataset))

    return no_dataset


def create_random_lists_multi(range_start, range_end, yousosu, multisu, randomkeisu, listsu=6):

    random_lists = []
    for cnt1 in range(listsu):
        cnt1 = cnt1 *randomkeisu
        tmp = []
        for cnt2 in range(multisu):
            random.seed(cnt1+cnt2+1)
            random_list = random.sample(range(range_start, range_end + 1), yousosu)
            tmp.append(random_list)
            #print(tmp)
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


def light_gbm(train_data, test_data):

    # 一列目のラベルを取得
    labels = [row[0] for row in train_data]

    # ラベルの個数をカウント
    label_counts = Counter(labels)

    # 一番個数が少ないラベルの数を取得
    min_label_count = min(label_counts.values())

    # 一番少ないラベルに合わせて他のラベルをフィルタリング
    balanced_data = []
    for label in label_counts.keys():
        filtered_data = [row for row in train_data if row[0] == label][:min_label_count]
        balanced_data.extend(filtered_data)

    # print("\nBalanced Data:")
    # for row in balanced_data:
    #     print(row)

    data = np.array(balanced_data)
    X = data[:,1:]
    print(X)
    y = data[:,0]-1
    print(y)

    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデル
    # max_depth=-1は無制限を意味する
    model = lgb.LGBMClassifier(force_col_wise=True, n_estimators=100, learning_rate=0.01, max_depth=-1 ,objective='multiclass')
    model.fit(X_train, y_train)

    # 評価
    score = model.score(X_val, y_val)
    print("score", score)

    # 推論
    predictions = sorted(list(map(int, set(model.predict(test_data)+1))))
    print("Predictions:", predictions)

    return score ,predictions


def light_gbm_nogood(train_data, test_data):

    # 一列目のラベルを取得
    labels = [row[0] for row in train_data]

    # ラベルの個数をカウント
    label_counts = Counter(labels)

    # 一番個数が少ないラベルの数を取得
    min_label_count = min(label_counts.values())

    # 一番少ないラベルに合わせて他のラベルをフィルタリング
    balanced_data = []
    for label in label_counts.keys():
        filtered_data = [row for row in train_data if row[0] == label][:min_label_count]
        balanced_data.extend(filtered_data)

    # print("\nBalanced Data:")
    # for row in balanced_data:
    #     print(row)

    data = np.array(balanced_data)
    X = data[:,1:]
    print(X)
    y = data[:,0]-1
    print(y)

    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデル
    # max_depth=-1は無制限を意味する
    model = lgb.LGBMClassifier(force_col_wise=True, n_estimators=1, learning_rate=1, max_depth=-1 ,objective='multiclass')
    model.fit(X_train, y_train)

    # 評価
    score = model.score(X_val, y_val)
    print("score", score)

    # 推論
    predictions = sorted(list(map(int, set(model.predict(test_data)+1))))
    print("Predictions:", predictions)

    return score ,predictions


def light_gbm_KFold(train_data, test_data):

    # 一列目のラベルを取得
    labels = [row[0] for row in train_data]
    
    # ラベルの個数をカウント
    label_counts = Counter(labels)

    # 一番個数が少ないラベルの数を取得
    min_label_count = min(label_counts.values())

    # 一番少ないラベルに合わせて他のラベルをフィルタリング
    balanced_data = []
    for label in label_counts.keys():
        filtered_data = [row for row in train_data if row[0] == label][:min_label_count]
        balanced_data.extend(filtered_data)

    # print("\nBalanced Data:")
    # for row in balanced_data:
    #     print(row)

    data = np.array(balanced_data)
    X = data[:,1:]
    print(X)
    y = data[:,0]-1
    print(y)

    FOLD = 3
    NUM_ROUND = 100
    # VERBOSE_EVAL = -1

    params = {
        'objective': 'regression',
        'verbose': -1,
        'num_leaves': 31,  # チューニングが必要
        'learning_rate': 0.1,  # チューニングが必要
    }

    valid_scores = []
    models = []
    kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=NUM_ROUND,
            # verbose_eval=VERBOSE_EVAL
            # early_stopping_rounds=50
        )

        y_valid_pred = model.predict(X_valid)
        score = mean_absolute_error(y_valid, y_valid_pred)
        print(f'fold {fold} MAE: {score}')
        valid_scores.append(score)

        models.append(model)
    
    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')

    # 推論
    # 最小値のインデックスを取得
    min_index = valid_scores.index(min(valid_scores))
    print("min_index",min_index)
    predictions = sorted(list(map(int, set(models[min_index].predict(test_data)+1))))
    print("Predictions:", predictions)

    return cv_score ,predictions


def light_gbm_multi(train_data, test_data):

    # 一列目のラベルを取得
    labels = [row[0] for row in train_data]

    # ラベルの個数をカウント
    label_counts = Counter(labels)

    # 一番個数が少ないラベルの数を取得
    min_label_count = min(label_counts.values())

    # 一番少ないラベルに合わせて他のラベルをフィルタリング
    balanced_data = []
    for label in label_counts.keys():
        filtered_data = [row for row in train_data if row[0] == label][:min_label_count]
        balanced_data.extend(filtered_data)

    # print("\nBalanced Data:")
    # for row in balanced_data:
    #     print(row)

    data = np.array(balanced_data)
    X = data[:,1:]
    print(X)
    y = data[:,0]-1
    print(y)

    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'multiclass',  # 多クラス分類を指定
        'num_class': 43,  # クラスの数を設定
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',  # 多クラスの対数尤度を使用
        'num_leaves': 32,
        'learning_rate': 0.1,
        'feature_fraction': 0.9
    }

    # LightGBMモデルを訓練（交差検証を使用）
    model = lgb.LGBMClassifier(**params, n_estimators=10)  # イテレーション回数はここで指定
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')  # 5分割交差検証

    # 交差検証スコアの平均を表示
    print(f'*** Cross-Validation Mean Accuracy: {np.mean(cv_scores)} ***')

    # 最終モデルをトレーニングデータ全体で訓練
    model.fit(X, y)

    # テストデータを予測
    y_pred = model.predict(X_val)

    # 正解率を計算
    accuracy = accuracy_score(y_val, y_pred)
    print(f'*** Accuracy on Validation Data: {accuracy} ***')

    # クラスごとの評価メトリクスを表示
    # report = classification_report(y_val, y_pred)
    # print(f'Classification Report:\n{report}')

    # 推論
    predictions = sorted(list(map(int, set(model.predict(test_data) + 1))))
    print("Predictions:", predictions)
    

    return accuracy, predictions

