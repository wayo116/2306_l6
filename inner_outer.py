import itertools
from combi import combi

def inner_outer(dlists,in_hani,out_hani,in_combisu,out_combisu):
    print('----inner_outer----')
    inner_list = list(set(list(itertools.chain.from_iterable(dlists[in_hani[0]:in_hani[1]]))))
    print('inner_list',inner_list)

    outer_list = list(set(list(itertools.chain.from_iterable(dlists[out_hani[0]:out_hani[1]]))))
    print('outer_list',outer_list)

    #outer_listに_inner_listを含まない
    outer_list_not_inner = sorted(list(set(outer_list) - set(inner_list)))
    print('outer_list_not_inner',outer_list_not_inner)
        
    inner_combi=combi(inner_list,in_combisu)
    print('inner組合せ数',len(inner_combi))
    outer_combi=combi(outer_list_not_inner,out_combisu)
    print('outer組合せ数',len(outer_combi))

    newlists=[]
    for inner in inner_combi:
        for outer in outer_combi:
            newlist=[]
            newlist = sorted(inner+outer)
            #print(newlist)
            newlists.append(newlist)
    print('inner outer組合せ数',len(newlists))

    return newlists

def inner_outer2(dlists,in_hani,out_hani,in_combisu,out_combisu,notinout_combisu):
    print('----inner_outer----')
    inner_list = list(set(list(itertools.chain.from_iterable(dlists[in_hani[0]:in_hani[1]]))))
    print('inner_list',inner_list)

    outer_list = list(set(list(itertools.chain.from_iterable(dlists[out_hani[0]:out_hani[1]]))))
    print('outer_list',outer_list)

    all_list = [i for i in range(1, 44)]
    print('all_list',all_list)

    #outer_listに_inner_listを含まない
    outer_list_not_inner = sorted(list(set(outer_list) - set(inner_list)))
    print('outer_list_not_inner',outer_list_not_inner)
        
    not_inout_list = sorted(list(set(all_list) - set(inner_list) - set(outer_list)))
    print('not_inout_list',not_inout_list)

    inner_combi=combi(inner_list,in_combisu)
    print('inner組合せ数',len(inner_combi))
    outer_combi=combi(outer_list_not_inner,out_combisu)
    print('outer組合せ数',len(outer_combi))
    notinout_combi=combi(not_inout_list,notinout_combisu)
    print('notinout_combi組合せ数',len(notinout_combi))

    newlists=[]
    for inner in inner_combi:
        for outer in outer_combi:
            for notinout in notinout_combi:
                newlist=[]
                newlist = sorted(inner+outer+notinout)
                #print(newlist)
                newlists.append(newlist)
    print('inner outer組合せ数',len(newlists))

    return newlists
