def tate_hani(dlists, kaisus):
    results=[]
    for ii in range(6):
        result=[]
        result = [dlists[kaisu][ii] for kaisu in kaisus]
        results.append(result)

    print(results)
