import itertools

def combi(num_list,combisu):
    print('----combi----')
    new_list = []
    for num in itertools.combinations(num_list, combisu):
        new_list.append(list(num))
    #print('new_list',new_list)
    return new_list
