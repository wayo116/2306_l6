#from datalists import dlists
import numpy as np
from matching import yobu
import itertools


def match_combi(dlists, kaisus, sikichi):
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

# kaisus = 1500
# sikichi = 10
# match_combi(dlists, kaisus, sikichi)
