# -*- coding: utf-8 -*-
"""main_v5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r4LSNABaTSmXD4XW2Eht0pbRUnen_EtK
"""

!git clone https://wayo116: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@github.com/wayo116/2306_l6.git

#!pip install scikit-image
# import tensorflow as tf
import os
# import collections,math
#import itertools
os.chdir('/content/2306_l6')

from datalists import dlists
# from Effi_v6 import Effi_TrainPred
# from randam_kaisai import randam_kaisai_multi3
# from find_kaisaihani import find_kaisaihani

from Dell6_v2 import Dell6
from inner_outer import inner_outer, inner_outer2
outlists=[]
saisinkekka_list=[18,25,26,37,39,43]
'''
print('\n----学習----')

model_name='V2B0'
#model_name='V2B1'
#model_name='V2B0_none'
#model_name='V2B1_none'
#model_name='V2B2_none'
#model_name='V2B3_none'
#model_name='RegY320'

kaisu=100 #100
setcount=1 #1
kaisu_pre=100
setcount_pre=1 #固定
kaisai_list=find_kaisaihani(dlists,[[0,100],[35,100],[18,150]],5)
kaisai_list_pre=find_kaisaihani(dlists,[[0,100],[35,100],[18,150]],5)
#kaisai_list=randam_kaisai_multi3(3,43,1,88,1,[[4,24],[35,50]],[0,0])
#kaisai_list_pre=randam_kaisai_multi3(3,43,1,55,1,[[4,24],[35,50]],[0,0])
pltflag=0
brendflag=0
inflated=35 #35
inflated_pre=1
kosuu=43 #固定
kosuu_pre=6
hw=96
rdmhani=1
compactness = 21
n_segments = 800
youso_list=randam_kaisai_multi3(3,43,1,55,1,[[0,5]],[0,0])

#baseimg_name='base_img' #黒地に緑数字 68*62
#baseimg_name='base_img2' #カラー43種 フリーサイズ
#baseimg_name='base_img2_2' #カラー43種 マスク 96*96
#baseimg_name='base_img3' #タイル1毎色違い43種 フリーサイズ
#baseimg_name='base_img4' #カラーver2 43種 フリーサイズ
#baseimg_name='base_img4_2' #カラーver2 色彩調整43種 224*224
#baseimg_name='base_img5' #カラー5種 サイズフリー
baseimg_name='base_img5_2' #色彩調整5種
#baseimg_name='base_img6' #タイル10毎色違い5種
#baseimg_name='base_img6_2' #タイル5毎色違い9種
#baseimg_name='base_img_rail' #スライドパズルレール43種

effi_trainpred = Effi_TrainPred(model_name,kaisu,kaisu_pre,kaisai_list,kaisai_list_pre,pltflag,brendflag,inflated,inflated_pre,hw,rdmhani,baseimg_name,setcount,setcount_pre,kosuu,kosuu_pre,compactness,n_segments,youso_list)

dlists=dlists
epochs=5 #5
batch_size=32

modelflag=0
if modelflag == 0:
    model = effi_trainpred.effi_train2(dlists,epochs,batch_size)
    model.save('./my_model.h5')

elif modelflag == 1:
    model = tf.keras.models.load_model('./my_model.h5')

print('\n----推論予想----')

select_box=3
index="None"
saisinkekka=saisinkekka_list
bunkatu=5
pred_dlists = effi_trainpred.pred(dlists,model,select_box,index)
outlist=Dell6(dlists, pred_dlists, saisinkekka, bunkatu).shori()
#outlists.extend(outlist)
#print('outlist',outlist)
'''
'''
print('\n----推論以外で予想----')

bunkatu=4
list_all = list(range(1,44))
diff_list = set(pred_dlists) ^ set(list_all)
outlist=Dell6(dlists, diff_list, saisinkekka, bunkatu).shori()
#outlists.extend(outlist)
#print('outlist',outlist)
'''

print('\n----インナーアウターで予想----')

saisinkekka=saisinkekka_list
bunkatu=5

in_hani=[0,1]
out_hani=[5,15]
in_combisu=1
out_combisu=5
pred_dlists=inner_outer(dlists,in_hani,out_hani,in_combisu,out_combisu)

#shori2は、pred_dlistsには組合せリストを入れる
outlist=Dell6(dlists, pred_dlists, saisinkekka, bunkatu).shori2()
#outlists.extend(outlist)
#print('outlist',outlist)

print('\n----インナーアウター2で予想----')

saisinkekka=saisinkekka_list
bunkatu=5

in_hani=[0,1]
out_hani=[5,15]
in_combisu=3
out_combisu=2
notinout_combisu=1
pred_dlists=inner_outer2(dlists,in_hani,out_hani,in_combisu,out_combisu,notinout_combisu)

#shori2は、pred_dlistsには組合せリストを入れる
outlist=Dell6(dlists, pred_dlists, saisinkekka, bunkatu).shori2()
#outlists.extend(outlist)
#print('outlist',outlist)