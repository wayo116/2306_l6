def tate_hani(dlists, kaisus):
    results=[]
    for ii in range(6):
        result=[]
        for kaisu in kaisus:
            result.append(dlists[kaisu][ii])
        results.append(result)

    print(results)