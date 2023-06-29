#from datalists import dlists
import numpy as np
from Utility.matching import yobu, aisho
import itertools


def yobu_combi(dlists, kaisus, sikichi):
    dcnt = yobu(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])

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
    dcnt = aisho(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])

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

    print(result)

    return result


def aisho_lists(dlists, kaisus, sikichi):
    dcnt = aisho(dlists, kaisus)

    result = []
    for youso in dlists[0]:
        result.append([idx+1 for idx, item in enumerate(dcnt[youso-1]) if item >= sikichi])

    print(result)

    return result


def yobu_dcnt(dlists, kaisus):
    
    return yobu(dlists, kaisus)


def aisho_dcnt(dlists, kaisus):
    
    return aisho(dlists, kaisus)

def find_elements(lst,st,ed):
    result = []
    for num in lst:
        if st <= num <= ed:
            result.append(num)
    print(lst)        
    print(result)
    return result
    
def yobu_aisho_combi(dlists, yobu_lists, aisho_dcnt, sikichi):

    result = []
    for ii in range(6):
        temp = []
        print("yobu_lists[" f"{ii}" "]=" f"{yobu_lists[ii]}")

        for youso in yobu_lists[ii]:
            print("youso",youso)
            print("aisho_dcnt[" f"{youso-1}" "]=" f"{aisho_dcnt[youso-1]}")
            temp.extend([idx+1 for idx, item in enumerate(aisho_dcnt[youso-1]) if item >= sikichi])
        print("temp",temp)

        sorted_result = sorted(set(temp)) 
        print("sorted_result",sorted_result)

        result.append(sorted_result)

    print("result",result)

    newlists=[]
    for no1 in find_elements(result[0],1,38):
        for no2 in find_elements(result[1],2,39):
            for no3 in find_elements(result[2],3,40):
                for no4 in find_elements(result[3],4,41):
                    for no5 in find_elements(result[4],5,42):
                        for no6 in find_elements(result[5],6,43):
                            newlist=[]
                            #if no1<39 and (no2>1 and no2<40) and (no3>2 and no2<41) and (no4>3 and no4<42) and (no5>4 and no5<43) and (no6>5 and no6<44): 
                            newlist = sorted([no1]+[no2]+[no3]+[no4]+[no5]+[no6])
                            if len(set(newlist)) == 6:
                                # print(newlist)
                                newlists.append(newlist)
    print('match_combi組合せ数',len(newlists))
    return newlists

