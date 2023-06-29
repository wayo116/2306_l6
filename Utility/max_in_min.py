def max_in_min(dcnts):
    result=[]
    for dcnt in dcnts:
        result.append(max(dcnt))
      
    return min(set(result))
