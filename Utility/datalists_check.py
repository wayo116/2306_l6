# from datalists import dlists
import numpy as np

def datalists_check(dlists, hani):
    # print(dlists[0:5])
    dlists = dlists[:hani]
    cntlist = [0] * (len(dlists) + 1)
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

    print("del_datalists",dlists[np.argmax(cntlist)])
    return dlists[np.argmax(cntlist)]

# datalists_check(dlists,10)