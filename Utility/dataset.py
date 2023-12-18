from datalists import dlists 
import random

import numpy as np
from collections import Counter

import csv
import os
import pandas as pd

#from sklearn.manifold import TSNE
#import umap
#from sklearn.decomposition import PCA

from scipy import stats


def no_dataset_trainval_multi(dlists, **dataset_params,):
    
    range_start = dataset_params["study_range_start"]
    range_end = dataset_params["study_range_end"]
    nmasi = dataset_params["study_nmasi"]
    bunseki_hani = dataset_params["bunseki_hani"]
    flat_hani = dataset_params["flat_hani"]
    z_thresh = dataset_params["z_thresh"]

    random.seed(42)  # 乱数のシードを42に設定
    shuffle_list = list(range(0, 6*flat_hani))
    random.shuffle(shuffle_list)
    #print("shuffle_list",shuffle_list)

    shokichi = 1
    no_dataset = []
    for kaisu, dlist in enumerate(dlists):
        if bunseki_hani >= flat_hani:
            kaisu_limit = len(dlists)-bunseki_hani
        else:
            kaisu_limit = len(dlists)-flat_hani
            
        # print("kaisu_limit",kaisu_limit)
        if kaisu >= kaisu_limit:
            break

        for dlist_retu in range(6):

            tmp = []
            tmp.append(dlist[dlist_retu])
            # tmp.append(int(dlists[kaisu+shokichi:kaisu+shokichi+1, dlist_retu]))

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
            # # パーセンタイル
            # percentile_25 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 25)
            # percentile_75 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 75)

            # # # 相関係数
            # # data = {
            # #     'X': [dlist[dlist_retu]]*bunseki_hani,
            # #     'Y': dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu]
            # # }
            # # df = pd.DataFrame(data)
            # # # 相関行列を計算
            # # correlation_matrix = df.corr()
            # # # 'X'と'Y'の相関係数を取得
            # # correlation_xy = correlation_matrix.loc['X', 'Y']
            # # # print("correlation_xy",correlation_xy)

            ## 列を削除した新しい2次元配列を生成
            #matrix = dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani]
            #column_to_remove = dlist_retu
            #new_matrix = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix]
            ## 全体の＊＊を計算
            #new_matrix_mean = np.mean(new_matrix)
            #new_matrix_var = np.var(new_matrix)
            #new_matrix_min = np.min(new_matrix)
            #new_matrix_max = np.max(new_matrix)

            # # 列を削除した新しい2次元配列を生成　ハーフ
            # matrix_h = dlists[kaisu+shokichi:kaisu+shokichi+int(bunseki_hani/2)]
            # column_to_remove = dlist_retu
            # new_matrix_h = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix_h]
            # # 全体の＊＊を計算 ハーフ
            # new_matrix_mean_h = np.mean(new_matrix_h)
            # new_matrix_var_h = np.var(new_matrix_h)
            # new_matrix_min_h = np.min(new_matrix_h)
            # new_matrix_max_h = np.max(new_matrix_h)

            # #tsne = TSNE(n_components = 2,perplexity=5) # n_componentsは低次元データの次元数
            # #X_tsne = tsne.fit_transform(matrix)
            
            # pca = PCA(n_components=1)
            # pca_result = pca.fit_transform(matrix)

            lists = list(range(1, bunseki_hani+1))
            values = lists[::-1]
            # 合計値を計算
            total = sum(values)
            # 各値を100%に対する割合に変換
            w = np.array([(value / total) * 100 for value in values])
            # w = np.array([5,4,3,2,1])
            kaju_ave = np.average(np.array(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu]),weights=w)

            # # st=1
            # # ed=18
            # # sted1 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # # st=10
            # # ed=24
            # # sted2 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # # st=35
            # # ed=50
            # # sted3 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # st=0
            # ed=50
            # sted4 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            ## 2次元配列を一次元配列に変換
            #two_dimensional_array = dlists[kaisu+shokichi:kaisu+shokichi+flat_hani]
            #flat_array = np.array(two_dimensional_array).flatten()
            ##print("flat_array",flat_array)
            ## 一次元配列をランダムに順番を変える
            #randomized_array = []
            ##for ii in shuffle_list:
            ##    randomized_array.append(flat_array[ii])
            #z = stats.zscore(flat_array)
            #z_abs = np.abs(z)
            #randomized_array.extend(z_abs)

            tmp.append(min_n)
            tmp.append(max_n)
            tmp.append(range_value)
            tmp.append(mean_n)
            tmp.append(sum_n)
            tmp.append(med_n)
            tmp.append(std_n)
            tmp.append(var_n)
            # tmp.append(percentile_25)
            # tmp.append(percentile_75)
            # # tmp.append(correlation_xy)

            #tmp.append(new_matrix_mean)
            #tmp.append(new_matrix_var)
            #tmp.append(new_matrix_min)
            #tmp.append(new_matrix_max)

            # tmp.append(new_matrix_mean_h)
            # tmp.append(new_matrix_var_h)
            # tmp.append(new_matrix_min_h)
            # tmp.append(new_matrix_max_h)
            # # tmp.extend(X_tsne[:,1])
            # tmp.extend(pca_result[:,0])

            tmp.append(kaju_ave)

            # #tmp.extend(sted1)
            # #tmp.extend(sted2)
            # #tmp.extend(sted3)
            # tmp.extend(sted4)

            #tmp.extend(randomized_array)

            tmp = [round(tmp[n], 2) for n in range(len(tmp))]
            
            if tmp != []:
                list1 = tmp[1:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start, range_end, yousosu=len(tmp), random_select=2, listsu=nmasi)

                list1_nmasi = []
                for list2 in list2s:
                    # print("list2",list2)
                    result = [x + y for x, y in zip(list1, list2)]
                    # print("result",result)
                    result.insert(0, dlist[dlist_retu])
                    # print("result_in",result)
                    list1_nmasi.append(result)
                # print("list1_nmasi",list1_nmasi)

            ###
            tmp_bool = []
            
            # 中央値＞平均
            med_dainari_mean = compe_bool(med_n, mean_n, None)
            # 最小値＞1
            min_dainari_1 = compe_bool(min_n, None, 1)
            # 43＞最大値
            max_shounari_43 = compe_bool(max_n, None, 43)
            # 文字含む
            moji = []
            for youso in dlists[kaisu+shokichi+1]:
                base_digits = decompose_to_digits(dlists[kaisu+shokichi, dlist_retu])
                # print("base_digits",base_digits)
                youso_digits = decompose_to_digits(youso)
                # print("youso_digits",youso_digits)
                # リストが一部含まれているかを判定
                if any(item in base_digits for item in youso_digits):
                    moji.append(1)
                else:
                    moji.append(0)
            # print("moji",moji)
                
            tmp_bool.append(med_dainari_mean)
            tmp_bool.append(min_dainari_1)
            tmp_bool.append(max_shounari_43)
            tmp_bool.extend(moji)

            list3_nmasi = [tmp_bool.copy() for _ in range(nmasi)]
            # print("list3_nmasi",list3_nmasi)

            for list1_3_nmasi in np.concatenate((np.array(list1_nmasi),np.array(list3_nmasi)), axis = 1):
                no_dataset.append(list1_3_nmasi.tolist())
                    
    no_dataset = remove_outliers(no_dataset, z_thresh)
    # print("no_dataset",no_dataset)
    print("no_dataset_rows",len(no_dataset))
    print("no_dataset_columns",len(no_dataset[0]))

    return no_dataset


def no_dataset_test_multi(dlists, **dataset_params):

    range_start = dataset_params["test_range_start"]
    range_end = dataset_params["test_range_end"]
    nmasi = dataset_params["test_nmasi"]
    bunseki_hani = dataset_params["bunseki_hani"]
    flat_hani = dataset_params["flat_hani"]
    test_dlists_hani = dataset_params["test_dlists_hani"]

    random.seed(42)  # 乱数のシードを42に設定
    shuffle_list = list(range(0, 6*flat_hani))
    random.shuffle(shuffle_list)
    #print("shuffle_list",shuffle_list)

    shokichi = 0
    no_dataset = []
    #for kaisu, dlist in enumerate(dlists[test_dlists_hani[0]:test_dlists_hani[1]]):
    for kaisu in test_dlists_hani:
        dlist = dlists[kaisu]
        if bunseki_hani >= flat_hani:
            kaisu_limit = len(dlists)-bunseki_hani
        else:
            kaisu_limit = len(dlists)-flat_hani
            
        # print("kaisu_limit",kaisu_limit)
        if kaisu >= kaisu_limit:
            break

        for dlist_retu in range(6):

            tmp = []
            # # tmp.append(dlist[dlist_retu])
            # tmp.append(int(dlists[kaisu+shokichi:kaisu+shokichi+1, dlist_retu]))

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
            # # パーセンタイル
            # percentile_25 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 25)
            # percentile_75 = np.percentile(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu], 75)

            # # # 相関係数
            # # data = {
            # #     'X': [dlist[dlist_retu]]*bunseki_hani,
            # #     'Y': dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu]
            # # }
            # # df = pd.DataFrame(data)
            # # # 相関行列を計算
            # # correlation_matrix = df.corr()
            # # # 'X'と'Y'の相関係数を取得
            # # correlation_xy = correlation_matrix.loc['X', 'Y']
            # # # print("correlation_xy",correlation_xy)

            ## 列を削除した新しい2次元配列を生成
            #matrix = dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani]
            #column_to_remove = dlist_retu
            #new_matrix = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix]
            # 全体の＊＊を計算
            #new_matrix_mean = np.mean(new_matrix)
            #new_matrix_var = np.var(new_matrix)
            #new_matrix_min = np.min(new_matrix)
            #new_matrix_max = np.max(new_matrix)

            # # 列を削除した新しい2次元配列を生成　ハーフ
            # matrix_h = dlists[kaisu+shokichi:kaisu+shokichi+int(bunseki_hani/2)]
            # column_to_remove = dlist_retu
            # new_matrix_h = [[row[i] for i in range(len(row)) if i != column_to_remove] for row in matrix_h]
            # # 全体の＊＊を計算 ハーフ
            # new_matrix_mean_h = np.mean(new_matrix_h)
            # new_matrix_var_h = np.var(new_matrix_h)
            # new_matrix_min_h = np.min(new_matrix_h)
            # new_matrix_max_h = np.max(new_matrix_h)

            # #tsne = TSNE(n_components = 2,perplexity=5) # n_componentsは低次元データの次元数
            # #X_tsne = tsne.fit_transform(matrix)
            
            # pca = PCA(n_components=1)
            # pca_result = pca.fit_transform(matrix)

            lists = list(range(1, bunseki_hani+1))
            values = lists[::-1]
            # 合計値を計算
            total = sum(values)
            # 各値を100%に対する割合に変換
            w = np.array([(value / total) * 100 for value in values])
            # w = np.array([5,4,3,2,1])
            kaju_ave = np.average(np.array(dlists[kaisu+shokichi:kaisu+shokichi+bunseki_hani, dlist_retu]),weights=w)

            # # st=1
            # # ed=18
            # # sted1 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # # st=10
            # # ed=24
            # # sted2 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # # st=35
            # # ed=50
            # # sted3 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            # st=0
            # ed=50
            # sted4 = dlists[kaisu+shokichi+st:kaisu+shokichi+ed, dlist_retu]

            ## 2次元配列を一次元配列に変換
            #two_dimensional_array = dlists[kaisu+shokichi:kaisu+shokichi+flat_hani]
            #flat_array = np.array(two_dimensional_array).flatten()
            ##print("flat_array",flat_array)
            ## 一次元配列をランダムに順番を変える
            #randomized_array = []
            ##for ii in shuffle_list:
            ##    randomized_array.append(flat_array[ii])
            #z = stats.zscore(flat_array)
            #z_abs = np.abs(z)
            #randomized_array.extend(z_abs)


            tmp.append(min_n)
            tmp.append(max_n)
            tmp.append(range_value)
            tmp.append(mean_n)
            tmp.append(sum_n)
            tmp.append(med_n)
            tmp.append(std_n)
            tmp.append(var_n)
            # tmp.append(percentile_25)
            # tmp.append(percentile_75)
            # # tmp.append(correlation_xy)

            #tmp.append(new_matrix_mean)
            #tmp.append(new_matrix_var)
            #tmp.append(new_matrix_min)
            #tmp.append(new_matrix_max)

            # tmp.append(new_matrix_mean_h)
            # tmp.append(new_matrix_var_h)
            # tmp.append(new_matrix_min_h)
            # tmp.append(new_matrix_max_h)
            # # tmp.extend(X_tsne[:,1])
            # tmp.extend(pca_result[:,0])

            tmp.append(kaju_ave)

            # #tmp.extend(sted1)
            # #tmp.extend(sted2)
            # #tmp.extend(sted3)
            # tmp.extend(sted4)
            
            #tmp.extend(randomized_array)
            
            tmp = [round(tmp[n], 2) for n in range(len(tmp))]
            
            if tmp != []:
                list1 = tmp[0:]
                # print("list1",list1)
                list2s = create_random_lists_float(range_start, range_end, yousosu=len(tmp), random_select=1, listsu=nmasi)

                list1_nmasi = []
                for list2 in list2s:
                    # print("list2",list2)
                    result = [x + y for x, y in zip(list1, list2)]
                    # print("result",result)
                    # result.insert(0, dlist[dlist_retu])
                    # print("result_in",result)
                    list1_nmasi.append(result)
                # print("list1_nmasi",list1_nmasi)

            ###
            tmp_bool = []
            
            # 中央値＞平均
            med_dainari_mean = compe_bool(med_n, mean_n, None)
            # 最小値＞1
            min_dainari_1 = compe_bool(min_n, None, 1)
            # 43＞最大値
            max_shounari_43 = compe_bool(max_n, None, 43)
            # 文字含む
            moji = []
            for youso in dlists[kaisu+shokichi+1]:
                base_digits = decompose_to_digits(dlists[kaisu+shokichi, dlist_retu])
                print("base_digits",base_digits)
                youso_digits = decompose_to_digits(youso)
                print("youso_digits",youso_digits)
                # リストが一部含まれているかを判定
                if any(item in base_digits for item in youso_digits):
                    moji.append(1)
                else:
                    moji.append(0)
            print("moji",moji)

            tmp_bool.append(med_dainari_mean)
            tmp_bool.append(min_dainari_1)
            tmp_bool.append(max_shounari_43)
            tmp_bool.extend(moji)

            list3_nmasi = [tmp_bool.copy() for _ in range(nmasi)]
            # print("list3_nmasi",list3_nmasi)

            for list1_3_nmasi in np.concatenate((np.array(list1_nmasi),np.array(list3_nmasi)), axis = 1):
                no_dataset.append(list1_3_nmasi.tolist())
            
    print("no_dataset",no_dataset)
    print("no_dataset_rows",len(no_dataset))
    print("no_dataset_columns",len(no_dataset[0]))

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


def create_random_lists_float(range_start, range_end, yousosu, random_select, listsu=6):
    
    random_lists = []
    for cnt in range(listsu):
        random.seed(cnt+1)
        if random_select == 1:
            random_list = [round(random.uniform(range_start, range_end), 2) for _ in range(yousosu)]
        if random_select == 2:
            random_list = [round(random.uniform(range_start, range_end),2)]*yousosu
        random_lists.append(random_list)

    # print("random_lists",random_lists)
    return random_lists


def remove_outliers(train_data, z_thresh):
    re_train_data = []
    train_data = np.array(train_data)
    # ラベル毎に、Zスコアが指定された閾値以上の列を含む外れ値を削除する
    for prelabel in range(43):
        prelabel_train_data = train_data[train_data[:, 0] == prelabel + 1, :]
        nolabel_train_data = prelabel_train_data[:,1:]
        #print("nolabel_train_data",nolabel_train_data)
        
        z_scores = np.abs((nolabel_train_data - np.mean(nolabel_train_data, axis=0)) / np.std(nolabel_train_data, axis=0))
        #print("z_scores",z_scores)
        outliers = np.any(z_scores > z_thresh, axis=1)
        #print("outliers",outliers)
        for one_array_data in prelabel_train_data[~outliers]:
            re_train_data.append(one_array_data.tolist())
    #print("re_train_data",re_train_data)
    return re_train_data


def compe_bool(value1, value2, thresh):
    if thresh == None:
        if value1 > value2:
            result_bool = 1
        else:
            result_bool = 0

    else:
        if value1 > thresh:
            result_bool = 1
        else:
            result_bool = 0

    # print("result_bool",result_bool)
    return result_bool


# 各数字を一桁ずつのリストに分解する関数
def decompose_to_digits(number):
    return [int(digit) for digit in str(number)]



