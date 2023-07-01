
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
        
        results.append(result)
        
    results2=[]
    for ii in range(6):
        result=[]
        c = collections.Counter(results[ii])
        counts = list(zip(*c.most_common()))
        countsx = counts[:topx]
        result = [min(countsx),max(countsx)]
        
        results2.append(result)
        
    print(results2)
    
    return results2
