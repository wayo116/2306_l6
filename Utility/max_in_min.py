def max_in_min(dcnts):
    result=[]
    for dcnt in dcnts:
        result.append(max(dcnt))
    print("閾値",min(set(result))) 
    return min(set(result))
