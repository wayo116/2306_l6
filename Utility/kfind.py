from datalists import dlists
from collections import Counter

st = 0
ed = 100
taisho_hani = 50
fbox_summary = []
for cnt in range(taisho_hani):
    print("st:"f"{st}" "-" "ed:"f"{ed}")
    new_dlists = dlists[st:ed]
    # print(new_dlists)

    print("new_dlists[0]:" f"{new_dlists[0]}")

    fbox = []
    for dyouso in new_dlists[0]:
        print("dyouso:" f"{dyouso}")

        for ii in range(1,len(new_dlists)):
            print(new_dlists[ii])
            if dyouso in new_dlists[ii]:
                print("hit-idx:" f"{ii}\n")
                fbox.append(ii)
                break
    print("fbox:"f"{fbox}\n")

    fbox_summary.append(fbox)

    st = st + 1
    ed = ed + 1

print(fbox_summary)

data = fbox_summary
# データをフラットなリストに変換
flat_data = [item for sublist in data for item in sublist]

# 数字の出現回数をカウント
counts = Counter(flat_data)

# 出現回数順にソート
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

# よく出現する数字の範囲を取得
frequent_numbers = [item[0] for item in sorted_counts[:10]]  # 出現回数の上位3つを取得

# 外れ値の範囲を取得
outlier_numbers = [item[0] for item in sorted_counts[-10:]]  # 出現回数の下位3つを取得

# 結果の表示
print("よく出現する数字の範囲:", min(frequent_numbers), "-", max(frequent_numbers))
print("外れ値の範囲:", min(outlier_numbers), "-", max(outlier_numbers))
