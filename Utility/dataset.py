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
import pandas as pd

#from sklearn.manifold import TSNE
#import umap
from sklearn.decomposition import PCA



def no_dataset_trainval_multi(dlists, **dataset_params,):

    range_start = dataset_params["study_range_start"]
    range_end = dataset_params["study_range_end"]
    nmasi = dataset_params["study_nmasi"]
    bunseki_hani = dataset_params["bunseki_hani"]

    shokichi = 1
    no_dataset = []
    for kaisu, dlist in enumerate(dlists):

        kaisu_limit = len(dlists)-100
        # print("kaisu_limit",kaisu_limit)
        if kaisu >= kaisu_limit:
            break

        for dlist_retu in range(6):

            tmp = []
            tmp.append(dlist[dlist_retu])
            tmp.append(int(dlists[kaisu+shokichi:kaisu+shokichi+1, dlist_retu]))

            # 最小
            min_n = np.min(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 最大
            max_n = np.max(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 範囲
            range_value = max_n - min_n
            # 平均
            mean_n = np.mean(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 合計
            sum_n = np.sum(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 中央
            med_n = np.median(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 標準偏差
            std_n = np.std(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 分散
            var_n = np.var(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # パーセンタイル
            percentile_25 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 25)
            percentile_75 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 75)

            # # 相関係数
            # data = {
            #     'X': [dlist[dlist_retu]]*bunseki_hani,
            #     'Y': dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu]
            # }
            # df = pd.DataFrame(data)
            # # 相関行列を計算
            # correlation_matrix = df.corr()
            # # 'X'と'Y'の相関係数を取得
            # correlation_xy = correlation_matrix.loc['X', 'Y']
            # # print("correlation_xy",correlation_xy)

            # 列を削除した新しい2次元配列を生成
            matrix = dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani]
            column_to_remove = dlist_retu
            new_matrix = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix]
            # 全体の＊＊を計算
            new_matrix_mean = np.mean(new_matrix)
            new_matrix_var = np.var(new_matrix)
            new_matrix_min = np.min(new_matrix)
            new_matrix_max = np.max(new_matrix)

            # 列を削除した新しい2次元配列を生成　ハーフ
            matrix_h = dlists[kaisu+shokichi:kaisu+shokichi+int(bunseki_hani/2)]
            column_to_remove = dlist_retu
            new_matrix_h = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix_h]
            # 全体の＊＊を計算 ハーフ
            new_matrix_mean_h = np.mean(new_matrix_h)
            new_matrix_var_h = np.var(new_matrix_h)
            new_matrix_min_h = np.min(new_matrix_h)
            new_matrix_max_h = np.max(new_matrix_h)

            #tsne = TSNE(n_components = 2,perplexity=5) # n_componentsは低次元データの次元数
            #X_tsne = tsne.fit_transform(matrix)
            
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(matrix)

            w = np.array([5,4,3,2,1])
            kaju_ave = np.average(np.array(dlists[kaisu+shokichi:kaisu+shokichi+5, dlist_retu]),weights=w)

            # st=1
            # ed=18
            # sted1 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # st=10
            # ed=24
            # sted2 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # st=35
            # ed=50
            # sted3 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            st=0
            ed=100
            sted4 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            tmp.append(min_n)
            tmp.append(max_n)
            tmp.append(range_value)
            tmp.append(mean_n)
            tmp.append(sum_n)
            tmp.append(med_n)
            tmp.append(std_n)
            tmp.append(var_n)
            tmp.append(percentile_25)
            tmp.append(percentile_75)
            # tmp.append(correlation_xy)

            tmp.append(new_matrix_mean)
            tmp.append(new_matrix_var)
            tmp.append(new_matrix_min)
            tmp.append(new_matrix_max)

            tmp.append(new_matrix_mean_h)
            tmp.append(new_matrix_var_h)
            tmp.append(new_matrix_min_h)
            tmp.append(new_matrix_max_h)
            # tmp.extend(X_tsne[:,1])
            tmp.extend(pca_result[:,0])

            tmp.append(kaju_ave)

            #tmp.extend(sted1)
            #tmp.extend(sted2)
            #tmp.extend(sted3)
            tmp.extend(sted4)

            if tmp != []:
                list1 = tmp[1:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start, range_end, yousosu=len(tmp), listsu=nmasi)
                
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


def no_dataset_test_multi(dlists, **dataset_params):

    range_start = dataset_params["test_range_start"]
    range_end = dataset_params["test_range_end"]
    nmasi = dataset_params["test_nmasi"]
    bunseki_hani = dataset_params["bunseki_hani"]
    test_dlists_hani = dataset_params["test_dlists_hani"]

    shokichi = 0
    no_dataset = []
    for kaisu, dlist in enumerate(dlists[test_dlists_hani[0]:test_dlists_hani[1]]):

        kaisu_limit = len(dlists)-100
        # print("kaisu_limit",kaisu_limit)
        if kaisu >= kaisu_limit:
            break

        for dlist_retu in range(6):

            tmp = []
            # tmp.append(dlist[dlist_retu])
            tmp.append(int(dlists[kaisu+shokichi:kaisu+shokichi+1, dlist_retu]))

            # 最小
            min_n = np.min(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 最大
            max_n = np.max(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 範囲
            range_value = max_n - min_n
            # 平均
            mean_n = np.mean(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 合計
            sum_n = np.sum(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 中央
            med_n = np.median(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 標準偏差
            std_n = np.std(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # 分散
            var_n = np.var(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu])
            # パーセンタイル
            percentile_25 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 25)
            percentile_75 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 75)

            # # 相関係数
            # data = {
            #     'X': [dlist[dlist_retu]]*bunseki_hani,
            #     'Y': dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu]
            # }
            # df = pd.DataFrame(data)
            # # 相関行列を計算
            # correlation_matrix = df.corr()
            # # 'X'と'Y'の相関係数を取得
            # correlation_xy = correlation_matrix.loc['X', 'Y']
            # # print("correlation_xy",correlation_xy)

            # 列を削除した新しい2次元配列を生成
            matrix = dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani]
            column_to_remove = dlist_retu
            new_matrix = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix]
            # 全体の＊＊を計算
            new_matrix_mean = np.mean(new_matrix)
            new_matrix_var = np.var(new_matrix)
            new_matrix_min = np.min(new_matrix)
            new_matrix_max = np.max(new_matrix)

            # 列を削除した新しい2次元配列を生成　ハーフ
            matrix_h = dlists[kaisu+shokichi:kaisu+shokichi+int(bunseki_hani/2)]
            column_to_remove = dlist_retu
            new_matrix_h = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix_h]
            # 全体の＊＊を計算 ハーフ
            new_matrix_mean_h = np.mean(new_matrix_h)
            new_matrix_var_h = np.var(new_matrix_h)
            new_matrix_min_h = np.min(new_matrix_h)
            new_matrix_max_h = np.max(new_matrix_h)

            #tsne = TSNE(n_components = 2,perplexity=5) # n_componentsは低次元データの次元数
            #X_tsne = tsne.fit_transform(matrix)
            
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(matrix)

            w = np.array([5,4,3,2,1])
            kaju_ave = np.average(np.array(dlists[kaisu+shokichi:kaisu+shokichi+5, dlist_retu]),weights=w)

            # st=1
            # ed=18
            # sted1 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # st=10
            # ed=24
            # sted2 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # st=35
            # ed=50
            # sted3 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            st=0
            ed=100
            sted4 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            tmp.append(min_n)
            tmp.append(max_n)
            tmp.append(range_value)
            tmp.append(mean_n)
            tmp.append(sum_n)
            tmp.append(med_n)
            tmp.append(std_n)
            tmp.append(var_n)
            tmp.append(percentile_25)
            tmp.append(percentile_75)
            # tmp.append(correlation_xy)

            tmp.append(new_matrix_mean)
            tmp.append(new_matrix_var)
            tmp.append(new_matrix_min)
            tmp.append(new_matrix_max)

            tmp.append(new_matrix_mean_h)
            tmp.append(new_matrix_var_h)
            tmp.append(new_matrix_min_h)
            tmp.append(new_matrix_max_h)
            # tmp.extend(X_tsne[:,1])
            tmp.extend(pca_result[:,0])

            tmp.append(kaju_ave)

            #tmp.extend(sted1)
            #tmp.extend(sted2)
            #tmp.extend(sted3)
            tmp.extend(sted4)

            if tmp != []:
                list1 = tmp[0:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start, range_end, yousosu=len(tmp), listsu=nmasi)
                
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
        #random_list = [round(random.uniform(range_start, range_end), 2) for _ in range(yousosu)]
        random_list = [round(random.uniform(range_start, range_end),2)]*yousosu
        random_lists.append(random_list)

    # print("random_lists",random_lists)
    return random_lists


def light_gbm(train_data, test_data, **lgbm_params):
    num_leaves = lgbm_params["num_leaves"]
    learning_rate = lgbm_params["learning_rate"]
    
    n_estimators = lgbm_params["n_estimators"]
    cv = lgbm_params["cv"]

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
    model = lgb.LGBMClassifier(force_col_wise=True, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=-1 ,objective='multiclass')
    model.fit(X_train, y_train)

    # 評価
    score = model.score(X_val, y_val)
    print("score", score)

    # 推論
    predictions = sorted(list(map(int, set(model.predict(test_data)+1))))
    print("Predictions:", predictions)

    return score ,predictions


def light_gbm_nogood(train_data, test_data, **lgbm_params):
    num_leaves = lgbm_params["num_leaves"]
    learning_rate = lgbm_params["learning_rate"]
    
    n_estimators = lgbm_params["n_estimators"]
    cv = lgbm_params["cv"]
    
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
    model = lgb.LGBMClassifier(force_col_wise=True, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=-1 ,objective='multiclass')
    model.fit(X_train, y_train)

    # 評価
    score = model.score(X_val, y_val)
    print("score", score)

    # 推論
    predictions = sorted(list(map(int, set(model.predict(test_data)+1))))
    print("Predictions:", predictions)

    return score ,predictions


def light_gbm_KFold(train_data, test_data, **lgbm_params):
    num_leaves = lgbm_params["num_leaves"]
    learning_rate = lgbm_params["learning_rate"]
    
    n_estimators = lgbm_params["n_estimators"]
    cv = lgbm_params["cv"]

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

    FOLD = cv
    NUM_ROUND = n_estimators
    # VERBOSE_EVAL = -1

    params = {
        'objective': 'regression',
        'verbose': -1,
        'num_leaves': num_leaves,  # チューニングが必要
        'learning_rate': learning_rate,  # チューニングが必要
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


def light_gbm_multi(train_data, test_data, **lgbm_params):
    num_leaves = lgbm_params["num_leaves"]
    learning_rate = lgbm_params["learning_rate"]
    
    n_estimators = lgbm_params["n_estimators"]
    cv = lgbm_params["cv"]

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
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': 0.9
    }

    # LightGBMモデルを訓練（交差検証を使用）
    model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)  # イテレーション回数はここで指定
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')  # 5分割交差検証

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


def light_gbm_v2(train_data, test_data, **lgbm_params):
    num_leaves = lgbm_params["num_leaves"]
    learning_rate = lgbm_params["learning_rate"]
    
    n_estimators = lgbm_params["n_estimators"]
    cv = lgbm_params["cv"]
    max_depth = lgbm_params["max_depth"]

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
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_val, y_val)

    params = {
        'objective': 'multiclass',  # 多クラス分類を指定
        'num_class': 43,  # クラスの数を設定
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',  # 多クラスの対数尤度を使用
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': 0.9,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
    }

    # LightGBMモデルを訓練（交差検証を使用）
    #model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)  # イテレーション回数はここで指定
    #model.fit(X_train, y_train)
    model = lgb.train(params,dtrain)

    # 評価
    score = model.score(dvalid)
    print("score", score)

    # 推論
    predictions = sorted(list(map(int, set(model.predict(test_data)+1))))
    print("Predictions:", predictions)

    return score ,predictions

