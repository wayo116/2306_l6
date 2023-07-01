#from datalists import dlists
import numpy as np
from Utility.matching import yobu, aisho, aisho_near
from Utility.tate_hani import tate_hani, tate_hani2
import itertools


def yobu_combi(dlists, kaisus, sikichi):
    dcnt = yobu(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])
        #result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi][0])

    print(result)

    newlists=[]
    for no1 in result[0]:
        for no2 in result[1]:
            for no3 in result[2]:
                for no4 in result[3]:
                    for no5 in result[4]:
                        for no6 in result[5]:
                            newlist=[]
                            newlist = sorted([no1]+[no2]+[no3]+[no4]+[no5]+[no6])
                            if len(set(newlist)) == 6:
                                # print(newlist)
                                newlists.append(newlist)
    print('match_combi組合せ数',len(newlists))
    return newlists


def aisho_combi(dlists, kaisus, sikichi):
    #dcnt = aisho(dlists, kaisus)
    dcnt = aisho_near(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])
        #result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi][0])

    print(result)

    newlists=[]
    for no1 in result[0]:
        for no2 in result[1]:
            for no3 in result[2]:
                for no4 in result[3]:
                    for no5 in result[4]:
                        for no6 in result[5]:
                            newlist=[]
                            newlist = sorted([no1]+[no2]+[no3]+[no4]+[no5]+[no6])
                            if len(set(newlist)) == 6:
                                # print(newlist)
                                newlists.append(newlist)
    print('match_combi組合せ数',len(newlists))
    return newlists


def yobu_lists(dlists, kaisus, sikichi):
    dcnt = yobu(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])
        #result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi][0])

    print(result)

    return result


def aisho_lists(dlists, kaisus, sikichi):
    #dcnt = aisho(dlists, kaisus)
    dcnt = aisho_near(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])
        #result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi][0])

    print(result)

    return result


def yobu_dcnt(dlists, kaisus):
    
    return yobu(dlists, kaisus)


def aisho_dcnt(dlists, kaisus):
    
    #return aisho(dlists, kaisus)
    return aisho_near(dlists, kaisus)


def find_elements(lst,st,ed):
    result = []
    for num in lst:
        if st <= num <= ed:
            result.append(num)
    #print(lst)        
    #print(result)
    return result


def yobu_aisho_combi(dlists, yobu_lists, aisho_dcnt, sikichi, thani):
    #tates = tate_hani(dlists,thani)
    tates = tate_hani2(dlists,thani,ttopx)
    
    result = []
    for ii in range(6):
        temp = []
        #print("yobu_lists[" f"{ii}" "]=" f"{yobu_lists[ii]}")

        for youso in yobu_lists[ii]:
            #print("youso",youso)
            #print("aisho_dcnt[" f"{youso-1}" "]=" f"{aisho_dcnt[youso-1]}")
            temp.extend([idx+1 for idx, item in enumerate(aisho_dcnt[youso-1]) if item >= sikichi])
            #temp.extend([idx+1 for idx, item in enumerate(aisho_dcnt[youso-1]) if item >= sikichi][0])
        #temp = find_elements(temp,ii+1,ii+38)
        #temp = find_elements(temp,min(tates[ii]),max(tates[ii]))
        temp = find_elements(temp,tates[ii][0],tates[ii][1])
        #print("temp",temp)

        sorted_result = sorted(set(temp)) 
        print("sorted_result",sorted_result)

        result.append(sorted_result)

    print("result",result)

    #new_result= []
    #new_result.append(result[0])
    #new_result.append(sorted(list(set(result[1])-set(result[0]))))
    #new_result.append(sorted(list(set(result[2])-set(result[1])-set(result[0]))))
    #new_result.append(sorted(list(set(result[3])-set(result[2])-set(result[1])-set(result[0]))))
    #new_result.append(sorted(list(set(result[4])-set(result[3])-set(result[2])-set(result[1])-set(result[0]))))
    #new_result.append(sorted(list(set(result[5])-set(result[4])-set(result[3])-set(result[2])-set(result[1])-set(result[0]))))
    #print("new_result",new_result)

    #result = new_result
    
    newlists=[]
    for no1 in result[0]:
        for no2 in result[1]:
            for no3 in result[2]:
                for no4 in result[3]:
                    for no5 in result[4]:
                        for no6 in result[5]:
                            newlist=[]
                            #if no1<39 and (no2>1 and no2<40) and (no3>2 and no2<41) and (no4>3 and no4<42) and (no5>4 and no5<43) and (no6>5 and no6<44): 
                            newlist = sorted([no1]+[no2]+[no3]+[no4]+[no5]+[no6])
                            if len(set(newlist)) == 6:
                                # print(newlist)
                                newlists.append(newlist)
    print('match_combi組合せ数',len(newlists))
    return newlists

