from PIL import Image
import numpy as np
import os,sys,glob

import subprocess
import itertools,random,collections
import matplotlib.pyplot as plt
from itertools import chain
import copy

from collections import Counter
from difflib import SequenceMatcher

class Dell6:
    def __init__(self, dlists, pred_dlists, saisinkekka, bunkatu):
        print('----Dell6:コンストラクタ----')
        self.dlists=dlists
        self.pred_dlists=pred_dlists
        self.saisinkekka=saisinkekka
        self.bunkatu=bunkatu
        print('dlists:{0}'.format(len(self.dlists)))
        print('pred_dlists:{0}'.format(len(self.pred_dlists)))
        
    def __del__(self):
        print("----Dell6:デストラクタ----")

    def combi(self, dlist1, sname='none'):
        outlist1=list(itertools.combinations(dlist1,6))
        print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        return outlist1

    def kako_most(self, dlist1, sname='none'): 
        outlist1=[]
        d1=list(itertools.chain.from_iterable(dlist1[:24,:]))
        c = collections.Counter(d1)
        for  ii in c.items():
            ii=list(ii)
            if ii[1]==3:
                outlist1.append(str(ii[0]))
        outlist1=[int(s) for  s in outlist1]
        print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        print(outlist1)
        return outlist1
        
    def kako_pm1(self, dlist1, sname='none'):
        outlist1=[]
        for d1 in dlist1[0,:]: 
            if d1==1:
                outlist1.append(d1+1)
            elif d1==43:
                outlist1.append(d1-1)
            else:
                outlist1.append(d1+1)
                outlist1.append(d1-1)
        print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        print(outlist1)
        return outlist1

    def kako_sm1(self, dlist1, sname='none'):
        outlist1=[]
        for d1 in dlist1[0,:]: 
            for  ii in range(-40, 41, 10):
                if ii!=0 and d1+ii>0 and d1+ii<44:
                    outlist1.append(d1+ii)
        outlist1=list(set(outlist1))
        print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        print(outlist1)
        return outlist1
        
    def kako_sum(self, dlist1, sname='none'):
        outlist1=[]
        sum1=0
        sgm=0.683/2
        for d1 in dlist1: 
            sum1=sum1+sum(d1)
        ave=sum1/len(dlist1)
        sgm_m=ave-(ave*sgm)
        outlist1.append(round(sgm_m))
        sgm_p=ave+(ave*sgm)
        outlist1.append(round(sgm_p))
        print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        print(outlist1)
        return outlist1

    def kako_minmax(self, dlist1, sname='none'):
        outlist1=[]
        for ii in range(6):
            outlist1.append(min(dlist1[:,ii]))
            outlist1.append(max(dlist1[:,ii]))
        print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        print(outlist1)
        return outlist1
        
    def kako_pattern(self, dlist1, sname='none'):
        #outlist0=[]
        outlist1=[]  
        for d1 in dlist1:
            #print(d1)
            #'''
            outlist0=[]
            if d1[4]>=37 and  d1[4]<=39 and d1[5]<=43 and d1[5]>=40:
                #print(d1)
                for ii in range(4):
                    if d1[ii]<10:
                        outlist0.append(1)
                    if d1[ii]>=10 and d1[ii]<20:
                        outlist0.append(10)
                    if d1[ii]>=20 and d1[ii]<30:
                        outlist0.append(20)
                    if d1[ii]>=30 and d1[ii]<40:
                        outlist0.append(30)
                outlist1.append(outlist0)
            else:
                outlist1.append([0,0,0,0])
        outlist1=list(set(map(tuple, outlist1)))
        #print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
        #print(outlist1)
        #'''
        return outlist1
        
    def check(self, dlist1, dlist2): 
        cnt3=0
        cnt4=0
        cnt5=0
        cnt6=0
        for d1 in dlist1: 
            if len(set(dlist2)&set(d1))==3:
                cnt3=cnt3+1
            if len(set(dlist2)&set(d1))==4:
                cnt4=cnt4+1
            if len(set(dlist2)&set(d1))==5:
                cnt5=cnt5+1
            if len(set(dlist2)&set(d1))==6:
                cnt6=cnt6+1
        print('3個当り{}'.format(cnt3))
        print('4個当り{}'.format(cnt4))
        print('5個当り{}'.format(cnt5))
        print('6個当り{}'.format(cnt6))
        print('予想個数{}'.format(len(dlist1)))
        print('予想当選額{0}'.format(-len(dlist1)*0.02+cnt3*0.1+cnt4*0.9+cnt5*30+cnt6*10000))

    def remove_duplicates(self,dlist,cnt,icchisu):
    
        # print("dlist[" f"{cnt}" "]:" f"{dlist[cnt]}\n")

        newlist = []
        newlist.extend(dlist[:cnt+1])

        for row in dlist[cnt+1:]:
            # print("row",row)
            
            if len(set(row) & set(dlist[cnt])) == icchisu: 
                newlist.append(row)

        # print("newlist:" f"{newlist}\n")

        return newlist

    def similar(self, dlist1):
        arr = dlist1
        cnt = 0
        icchisu = 3

        while True:

            # print("cnt",cnt)

            if cnt == len(arr):
                break

            arr = self.remove_duplicates(arr,cnt,icchisu)

            cnt = cnt + 1
        #dlist1 = arr
        print("len(arr)",len(arr))

        return arr
        
    def notkako_xdel(self, dlist1, dlist2, ifpattern, sname='none'):
        outlist1=[]
        d1=dlist1
        for d2 in dlist2: 
            if ifpattern==1 and len(set(d1)&set(d2))<4 and len(set(d1)&set(d2))>=0:
                outlist1.append(d2)
            if ifpattern==2 and len(set(d1)&set(d2))<=3:
                outlist1.append(d2)
            if ifpattern==3:
                cnt=0
                for ii in range(5):
                    if d2[ii]+1==d2[ii+1]:
                        cnt=cnt+1
                if cnt<=2:
                    outlist1.append(d2)
            if ifpattern==4 and len(set(d1)&set(d2))==1:
                outlist1.append(d2)
            if ifpattern==5 and d1[0]<sum(d2) and d1[1]>sum(d2):
                outlist1.append(d2)
            if ifpattern==6:
                cnt=0
                for ii in range(6):
                    if d1[ii*2]<=d2[ii] and d2[ii]<=d1[ii*2+1]:
                        cnt=cnt+1
                if cnt==6:
                    outlist1.append(d2)
            if ifpattern==7:
                cnt=0
                for ii in range(4):
                    if d2[ii]+1==d2[ii+1] and d2[ii]+2==d2[ii+2] :
                        cnt=cnt+1
                for ii in range(3):
                    if d2[ii]+1==d2[ii+1] and d2[ii]+2==d2[ii+2] and d2[ii]+3==d2[ii+3] :
                        cnt=cnt+1
                for ii in range(2):
                    if d2[ii]+1==d2[ii+1] and d2[ii]+2==d2[ii+2] and d2[ii]+3==d2[ii+3] and d2[ii]+4==d2[ii+4]:
                        cnt=cnt+1
                for ii in range(1):
                    if d2[ii]+1==d2[ii+1] and d2[ii]+2==d2[ii+2] and d2[ii]+3==d2[ii+3] and d2[ii]+4==d2[ii+4] and d2[ii]+5==d2[ii+5]:
                        cnt=cnt+1
                if cnt==0:
                    outlist1.append(d2)
            if ifpattern==8:
                cnt=0
                for ii in range(6):
                    if d2[ii]%2==0:
                        cnt=cnt+1
                if cnt>=2 and cnt<=4:
                    outlist1.append(d2)
            if ifpattern==9:
                for d1 in dlist1:
                    d22=self.kako_pattern([d2], '過去数字、パターン')
                    #print(list(d22[0]))
                    if list(d1)==list(d22[0]):
                        outlist1.append(d2)
            if ifpattern==10 and len(set(d1)&set(d2))>=2:
                outlist1.append(d2)
        if len(outlist1)>0:       
            print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(outlist1), type(outlist1)))
            return outlist1
        else:
            print('{0} outlist1>>len:{1} type:{2}'.format(sname, len(dlist2), type(dlist2)))
            return dlist2

    # def get_outlist(self, dlist1, sname='none'): 
    #     print('dlist1>>len:{0}'.format(len(dlist1)))
    #     outlist1=[]
    #     for  ii in range(len(dlist1)-1):
    #         #'print('dlist1-ii[{0}]:{1}'.format(ii, sorted(dlist1[ii])))
    #         for  ii2 in range(ii+1, len(dlist1)):
    #             #print('dlist1-ii2[{0}]:{1}'.format(ii2, sorted(dlist1[ii2])))
    #             if 4<len(set(sorted(dlist1[ii]))-set(sorted(dlist1[ii2]))):
    #                 outlist1.append(sorted(dlist1[ii2]))
    #                 #print('dlist1-ii2[{0}]:{1}'.format(ii2, sorted(dlist1[ii2])))
    #                 #print('set-:{0}'.format(set(sorted(dlist1[ii]))-set(sorted(dlist1[ii2]))))
    #     print('outlist1>>len:{0}'.format(len(list(set(map(tuple, outlist1))))))
    #     for d1 in list(set(map(tuple, outlist1))):
    #         print('{0} get_outlist:{1}'.format(sname, d1))
    #     return len(list(set(map(tuple, outlist1))))      	  

    def get_outlist2(self, dlist1, sname='none'):

        print('dlist1>>len:{0}'.format(len(dlist1)))
        outlist1=[]
        bunkatu=self.bunkatu
        k=round(len(dlist1)/(bunkatu+1))
        for ii in range(k, len(dlist1)-k, k):
            print('-----{}-----'.format(ii))
            for ii2 in range(1):
                 outlist1.append(dlist1[ii+ii2])
                 print('{0} get_outlist:{1}'.format(sname, dlist1[ii+ii2]))
        return outlist1
        
    # def shori(self):
    #     outlist1=self.combi(self.pred_dlists, '組合せ')
    #     outlist1=self.notkako_xdel(self.kako_sum(self.dlists, '合計範囲'), outlist1, 5, '合計範囲内')
    #     outlist1=self.notkako_xdel(self.kako_minmax(self.dlists[:24,:], '最小最大'), outlist1, 6, '最小最大内')
    #     outlist1=self.notkako_xdel(0, outlist1, 3, '連番、0〜2個含む')
    #     #outlist1=self.notkako_xdel(0, outlist1, 7, '3〜6連番、含まない')
    #     #outlist1=self.notkako_xdel(0, outlist1, 8, '偶数2〜4個、含む')
    #     outlist1=self.notkako_xdel(list(itertools.chain.from_iterable(self.dlists[:3,:])), outlist1, 1 ,'過去3回数字、0〜3個含む') 
    #     outlist1=self.notkako_xdel(self.dlists[0,:], outlist1, 2, '前回数字、0〜3個含む') 
    #     #outlist1=self.notkako_xdel(list(range(40, 44, 1)), outlist1, 4, '40〜43含む')
    #     #outlist1=self.notkako_xdel(list(range(41, 43, 1)), outlist1, 4, '42含む')      
    #     #outlist1=self.notkako_xdel(list(range(30, 40, 1)), outlist1, 4, '30〜39、1個含む')
    #     outlist1=self.notkako_xdel(list(range(1, 10, 1)), outlist1, 4, '1〜9、1個含む')
    #     #outlist1=self.notkako_xdel(list(range(10, 20, 1)), outlist1, 4, '10〜19、1個含む')
    #     #outlist1=self.notkako_xdel(list(range(20, 30, 1)), outlist1, 4, '20〜29、1個含む')
    #     #outlist1=self.notkako_xdel(self.kako_pm1(self.dlists, '前回数字±1'), outlist1, 10, '前回数字±1、2個以上含む')
    #     #outlist1=self.notkako_xdel(self.kako_sm1(self.dlists, '前回数字下1桁'), outlist1, 10, '前回数字下1桁、2個以上含む')
    #     #outlist1=self.notkako_xdel(self.kako_most(self.dlists, '過去24回数字、3回出現'), outlist1, 10, '過去24回数字、3回出現、2個以上含む')
    #     #outlist1=self.notkako_xdel(self.kako_pattern(self.dlists[:,:], '過去数字、パターン'), outlist1, 9, '過去数字、パターン、含む')
        
    #     #dlen=self.get_outlist(outlist1, '準重複削除')

    #     self.check(outlist1, self.saisinkekka)

    #     outlist2=self.get_outlist2(outlist1, '**') 
    #     self.check(outlist2, self.saisinkekka)
    #     return outlist2

    def shori2(self):
        outlist1=self.pred_dlists
        outlist1=self.notkako_xdel(self.kako_sum(self.dlists, '合計範囲'), outlist1, 5, '合計範囲内')
        outlist1=self.notkako_xdel(self.kako_minmax(self.dlists[:24,:], '最小最大'), outlist1, 6, '最小最大内')
        outlist1=self.notkako_xdel(0, outlist1, 3, '連番、0〜2個含む')
        outlist1=self.notkako_xdel(0, outlist1, 7, '3〜6連番、含まない')
        outlist1=self.notkako_xdel(0, outlist1, 8, '偶数2〜4個、含む')
        outlist1=self.notkako_xdel(list(itertools.chain.from_iterable(self.dlists[:3,:])), outlist1, 1 ,'過去3回数字、0〜3個含む') 
        outlist1=self.notkako_xdel(self.dlists[0,:], outlist1, 2, '前回数字、0〜3個含む') 
        #outlist1=self.notkako_xdel(list(range(40, 44, 1)), outlist1, 4, '40〜43含む')
        #outlist1=self.notkako_xdel(list(range(41, 43, 1)), outlist1, 4, '42含む')      
        #outlist1=self.notkako_xdel(list(range(30, 40, 1)), outlist1, 4, '30〜39、1個含む')
        outlist1=self.notkako_xdel(list(range(1, 10, 1)), outlist1, 4, '1〜9、1個含む')
        #outlist1=self.notkako_xdel(list(range(10, 20, 1)), outlist1, 4, '10〜19、1個含む')
        #outlist1=self.notkako_xdel(list(range(20, 30, 1)), outlist1, 4, '20〜29、1個含む')
        outlist1=self.notkako_xdel(self.kako_pm1(self.dlists, '前回数字±1'), outlist1, 10, '前回数字±1、2個以上含む')
        outlist1=self.notkako_xdel(self.kako_sm1(self.dlists, '前回数字下1桁'), outlist1, 10, '前回数字下1桁、2個以上含む')
        outlist1=self.notkako_xdel(self.kako_most(self.dlists, '過去24回数字、3回出現'), outlist1, 10, '過去24回数字、3回出現、2個以上含む')
        outlist1=self.notkako_xdel(self.kako_pattern(self.dlists[:,:], '過去数字、パターン'), outlist1, 9, '過去数字、パターン、含む')
        
        #dlen=self.get_outlist(outlist1, '準重複削除')

        self.check(outlist1, self.saisinkekka)

        outlist1 = self.similar(outlist1)
        self.check(outlist1, self.saisinkekka)
        
        outlist2=self.get_outlist2(outlist1, '**') 
        self.check(outlist2, self.saisinkekka)
        return outlist2
