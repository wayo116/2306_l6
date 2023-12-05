# from datalists import dlists
import numpy as np

def datalists_check(dlists, hani, kosu):
    # print(dlists[0:5])
    dlists = dlists[:hani]
    cntlist = [0] * (len(dlists) + 1)
    cntlists =[]
    for kaisu in range(len(dlists)):
        # print("kaisu",kaisu)

        for kaisu_ikou in range(kaisu+1,len(dlists)):
            # print("kaisu_ikou",kaisu_ikou)
            nankaimae = kaisu_ikou - kaisu
            # print("nankaimae",nankaimae)

            if dlists[kaisu] not in dlists[kaisu_ikou]:
                # print("nankaimae",nankaimae)
                # print("dlists[kaisu]",dlists[kaisu])
                # print("dlists[kaisu_ikou]",dlists[kaisu_ikou])
                cntlist[nankaimae] = cntlist[nankaimae] + 1

    #print("del_datalists",dlists[np.argmax(cntlist)])
    #return dlists[np.argmax(cntlist)]
    for n in range(1,kosu+1):
        idx = np.where(cntlist==np.sort(cntlist)[-n])[0]
        print("idx",idx[0])
        print("dlists",dlists[idx[0]])
        cntlists.extend(dlists[idx[0]])
    print("cntlists",cntlists)
    return cntlists
    

# datalists_check(dlists,10)")
    return cntlists
    

# datalists_check(dlists,10)
