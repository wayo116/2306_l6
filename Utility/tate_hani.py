
import collections

def tate_hani(dlists, kaisus):
    results=[]
    for ii in range(6):
        result=[]
        result = [dlists[kaisu][ii] for kaisu in range(kaisus)]
        results.append(result)

    print(results)
    
    return results

def tate_hani2(dlists, kaisus, topx):
    results=[]
    for ii in range(6):
        result=[]
        result = [dlists[kaisu][ii] for kaisu in range(kaisus)]
        counts = zip(*c.most_common())
        counts = counts[topx]
        result = [min(counts),max(counts)]
        results.append(result)

    print(results)
    
    return results
