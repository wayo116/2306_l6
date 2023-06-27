from datalists import dlists
import numpy as np

def yobu(dlists, kaisus):
    dcnt =  np.zeros((43,43))

    # kaisus = 100
    for kaisu in range(kaisus):
        for youso in range(6):
            mae_kaisu = kaisu
            usiro_kaisu = kaisu + 1

            result = dcnt[int(dlists[mae_kaisu][youso]-1), int(dlists[usiro_kaisu][youso]-1)]
            result = int(result + 1)
            dcnt[int(dlists[mae_kaisu][youso]-1),int(dlists[usiro_kaisu][youso]-1)] = result

    np.set_printoptions(threshold=np.inf)
    print("dcnt",dcnt)

    return dcnt


def aisho(dlists, kaisus):
    dcnt =  np.zeros((43,43))

    # kaisus = 100
    for kaisu in range(kaisus):
        for youso in range(6):
            for youso_hoka in range(6):
                if dlists[kaisu][youso] != dlists[kaisu][youso_hoka]:

                    result = dcnt[int(dlists[kaisu][youso]-1), int(dlists[kaisu][youso_hoka]-1)]
                    result = int(result + 1)
                    dcnt[int(dlists[kaisu][youso]-1), int(dlists[kaisu][youso_hoka]-1)] = result

    np.set_printoptions(threshold=np.inf)
    print("dcnt",dcnt)

    return dcnt

#kaisus = 1500
#yobu(dlists, kaisus)
#aisho(dlists, kaisus)

        
