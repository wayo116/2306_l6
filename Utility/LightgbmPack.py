import os
import csv
import time
import numpy as np
import random

from datalists import dlists
from Utility.dataset import no_dataset_trainval_multi, no_dataset_test_multi

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LightgbmPack():
    def __init__(self):
        print("LightgbmPack")
        
    def lightgbmpack(self, kaisai, saisinkekka_list, dlists, dlists_end, **params):
        print('\n----lightGBMで予想----')
        start = time.time()

        dataset_params = params["dataset_params"]
        lgbm_model = params["lgbm_model"]
        lgbm_params = params["lgbm_params"]

        #学習検証用
        train_val_dlists = dlists[1:dlists_end]
        train_data = no_dataset_trainval_multi(train_val_dlists, **dataset_params)
        print("len(train_data)",len(train_data))

        #テスト用
        #test_data=[]
        #for value in dlists[0]:
        #    cnt=0
        #    for row in train_data:
        #        if cnt < 3:
        #            if row[0] == value:
        #                test_data.append(row[1:])
        #                cnt=cnt+1

        #テスト用
        test_dlists = dlists[0:dlists_end]
        test_data = no_dataset_test_multi(test_dlists, **dataset_params)
        print("len(test_data)",len(test_data))

        #lightgbmで推論
        print("model_type",lgbm_model["model_type"])
        if lgbm_model["model_type"] == "light_gbm":
            score ,predictions = self.light_gbm(train_data, test_data, **lgbm_params)
            
        if lgbm_model["model_type"] == "light_gbm_optuna":
            score ,predictions = self.light_gbm_optuna(train_data, test_data, **lgbm_params)

        self.saisinkekkalist_predictions_chunk(saisinkekka_list, predictions, dataset_params["test_nmasi"])
        
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

    
    def saisinkekkalist_predictions_chunk(self, saisinkekka_list, predictions, chunk_size):
        
        chunks = [predictions[i:i + chunk_size] for i in range(0, len(predictions), chunk_size)]

        result1 = []
        result2 = []
        result3 = []
        for chunk in chunks:
            # print(chunk)
            # result.append(any(element in saisinkekka_list for element in chunk))
            result1.append(list(set(saisinkekka_list) & set(chunk)))
            result2.append(len(set(saisinkekka_list) & set(chunk)))

            # リストの要素をカウント
            chunk_counts = Counter(chunk)
            # 上位n件の要素と出現回数を取得
            top_n = chunk_counts.most_common(2)
            for element, count in top_n:
                result3.append(element)
                
        print("result1",result1)
        print("result2",result2)
        sorted_result3 = sorted(list(set(result3)))
        print("sorted_result3",sorted_result3)

        
    def light_gbm(self, train_data, test_data, **lgbm_params):
        import lightgbm as lgb
        
        objective = lgbm_params["objective"]
        num_class = lgbm_params["num_class"]
        boosting_type = lgbm_params["boosting_type"]
        metric = lgbm_params["metric"]
        num_leaves = lgbm_params["num_leaves"]
        learning_rate = lgbm_params["learning_rate"]
        feature_fraction = lgbm_params["feature_fraction"]
        max_depth = lgbm_params["max_depth"]
        random_seed = lgbm_params["random_seed"]
        force_row_wise = lgbm_params["force_row_wise"]
        feature_pre_filter = lgbm_params["feature_pre_filter"]
        lambda_l1 = lgbm_params["lambda_l1"]
        lambda_l2 = lgbm_params["lambda_l2"]
        bagging_fraction = lgbm_params["bagging_fraction"]
        bagging_freq = lgbm_params["bagging_freq"]
        min_data_in_leaf = lgbm_params["min_data_in_leaf"]
        num_boost_round = lgbm_params["num_boost_round"]
        verbose = lgbm_params["verbose"]
    
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
        print(len(X))
        y = data[:,0]-1
        print(y)
        print(len(y))
    
        # データ分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_val, y_val)
    
        params = {
            'objective': objective,
            'num_class': num_class,
            'boosting_type': boosting_type,
            'metric': metric,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'max_depth': max_depth,
            'random_seed': random_seed,
            'force_row_wise': force_row_wise,
            'feature_pre_filter': feature_pre_filter,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_data_in_leaf': min_data_in_leaf,
            'num_boost_round': num_boost_round,
            'verbose': verbose,
        }
    
        # LightGBMモデルを訓練
        model = lgb.train(params,dtrain, valid_sets=[dtrain, dvalid],)
    
        # 評価
        # score = model.score(dvalid)
        # print("score", score)
        preds_X_val = model.predict(X_val)
        preds_X_val = np.argmax(preds_X_val, axis=1) + 1 # 予測結果のクラスの値を調整
        accuracy = accuracy_score(y_val+1, preds_X_val)
        print("accuracy",accuracy)
    
        # 推論
        # predictions = sorted(list(map(int, set(model.predict(test_data)+1))))
        # print("Predictions:", predictions)
        predictions = model.predict(test_data)
        #print("predictions",predictions)
        
        predictions = np.argmax(predictions, axis=1) + 1 # 予測結果のクラスの値を調整
        print("predictions",predictions)
    
        return accuracy ,predictions


    def light_gbm_optuna(self, train_data, test_data, **lgbm_params):
        import optuna.integration.lightgbm as lgb
    
        objective = lgbm_params["objective"]
        num_class = lgbm_params["num_class"]
        boosting_type = lgbm_params["boosting_type"]
        metric = lgbm_params["metric"]
        num_leaves = lgbm_params["num_leaves"]
        learning_rate = lgbm_params["learning_rate"]
        feature_fraction = lgbm_params["feature_fraction"]
        max_depth = lgbm_params["max_depth"]
        random_seed = lgbm_params["random_seed"]
        force_row_wise = lgbm_params["force_row_wise"]
        feature_pre_filter = lgbm_params["feature_pre_filter"]
        lambda_l1 = lgbm_params["lambda_l1"]
        lambda_l2 = lgbm_params["lambda_l2"]
        bagging_fraction = lgbm_params["bagging_fraction"]
        bagging_freq = lgbm_params["bagging_freq"]
        min_data_in_leaf = lgbm_params["min_data_in_leaf"]
        num_boost_round = lgbm_params["num_boost_round"]
        verbose = lgbm_params["verbose"]
    
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
        print(len(X))
        y = data[:,0]-1
        print(y)
        print(len(y))
    
        # データ分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        dtrain = lgb.Dataset(X_train, y_train)
        dvalid = lgb.Dataset(X_val, y_val)
    
        params = {
            'objective': objective,
            'num_class': num_class,
            'boosting_type': boosting_type,
            'metric': metric,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'max_depth': max_depth,
            'random_seed': random_seed,
            'force_row_wise': force_row_wise,
            'feature_pre_filter': feature_pre_filter,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_data_in_leaf': min_data_in_leaf,
            'num_boost_round': num_boost_round,
            'verbose': verbose,
        }
    
        # LightGBMモデルを訓練
        model = lgb.train(params,dtrain, valid_sets=[dtrain, dvalid],)
        print("best params", model.params)
        accuracy = 0
        predictions = []
        
        return accuracy ,predictions
