from datalists import dlists
from collections import Counter

def kfind(dlist, taisho_hani):
    st = 0
    ed = 100
    # taisho_hani = 50
    fbox_summary = []
    for cnt in range(taisho_hani):
        # print("st:"f"{st}" "-" "ed:"f"{ed}")
        new_dlists = dlists[st:ed]
        # print(new_dlists)

        # print("new_dlists[0]:" f"{new_dlists[0]}")

        fbox = []
        for dyouso in new_dlists[0]:
            # print("dyouso:" f"{dyouso}")

            for ii in range(1,len(new_dlists)):
                # print(new_dlists[ii])

                if dyouso in new_dlists[ii]:
                    # print("hit-idx:" f"{ii}\n")
                    fbox.append(ii)
                    break

        # print("fbox:"f"{fbox}\n")

        fbox_summary.append(fbox)

        st = st + 1
        ed = ed + 1

    # print(fbox_summary)

    data = fbox_summary
    # データをフラットなリストに変換
    flat_data = [item for sublist in data for item in sublist]

    # 数字の出現回数をカウント
    counts = Counter(flat_data)

    # 出現回数順にソート
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # よく出現する数字の範囲を取得
    frequent_numbers = [item[0] for item in sorted_counts[:10]]  # 出現回数の上位*つを取得

    # 外れ値の範囲を取得
    outlier_numbers = [item[0] for item in sorted_counts[-10:]]  # 出現回数の下位*つを取得

    # 結果の表示
    print("よく出現する数字:", sorted_counts)
    # print("よく出現する数字の範囲:", min(frequent_numbers), "-", max(frequent_numbers))
    # print("外れ値の範囲:", min(outlier_numbers), "-", max(outlier_numbers))

    inner_list = []
    outer_list = []
    inner_list.append(min(frequent_numbers)-1)
    inner_list.append(max(frequent_numbers)-1)
    outer_list.append(max(frequent_numbers)-1)
    outer_list.append(max(outlier_numbers)-1)
    print("inner_list:"f"{inner_list}\n" "outer_list:"f"{outer_list}\n")

    return inner_list, outer_list


def kfind2(dlist, taisho_hani):
    st = 0
    ed = 100
    # taisho_hani = 50
    fbox_summary = []
    for cnt in range(taisho_hani):
        # print("st:"f"{st}" "-" "ed:"f"{ed}")
        new_dlists = dlists[st:ed]
        # print(new_dlists)

        # print("new_dlists[0]:" f"{new_dlists[0]}")

        fbox = []
        for dyouso in new_dlists[0]:
            # print("dyouso:" f"{dyouso}")

            for ii in range(1,len(new_dlists)):
                # print(new_dlists[ii])

                if dyouso in new_dlists[ii]:
                    # print("hit-idx:" f"{ii}\n")
                    fbox.append(ii)
                    break

        # print("fbox:"f"{fbox}\n")

        fbox_summary.append(fbox)

        st = st + 1
        ed = ed + 1

    # print(fbox_summary)

    data = fbox_summary
    # データをフラットなリストに変換
    flat_data = [item for sublist in data for item in sublist]

    # 数字の出現回数をカウント
    counts = Counter(flat_data)

    # 出現回数順にソート
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # よく出現する数字の範囲を取得
    # frequent_numbers = [item[0] for item in sorted_counts[:10]]  # 出現回数の上位*つを取得
    frequent_numbers = [item[0] for item in sorted_counts[:10] if item[1]>1]

    # 外れ値の範囲を取得
    # outlier_numbers = [item[0] for item in sorted_counts[-10:]]  # 出現回数の下位*つを取得
    outlier_numbers = [item[0] for item in sorted_counts[:10] if item[1]<2]

    # 結果の表示
    print("よく出現する数字:", sorted_counts)
    # print("よくoutlier_numbers出現する数字の範囲:", min(frequent_numbers), "-", max(frequent_numbers))
    # print("外れ値の範囲:", min(outlier_numbers), "-", max(outlier_numbers))

    # inner_list = []
    # outer_list = []
    # inner_list.append(min(frequent_numbers)-1)
    # inner_list.append(max(frequent_numbers)-1)
    # outer_list.append(max(frequent_numbers)-1)
    # outer_list.append(max(outlier_numbers)-1)

    inner_list = frequent_numbers
    outer_list = outlier_numbers
    print("inner_list:"f"{inner_list}\n" "outer_list:"f"{outer_list}\n")

    return inner_list, outer_list


# inner_list, outer_list = kfind(dlists,20)

