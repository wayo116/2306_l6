def per_hyouji(saisin,pred)
    s_and_p = set(saisin) & set(pred)
    s_and_p_len = len(s_and_p)
    pred_len = len(pred)
    
    if s_and_p_len > 0 and pred_len > 0:
        percent = round(s_and_p_len/pred_len*100)
        print(f"{s_and_p_len}/{pred_len}")
    else:
        percent = 0
    print("percent",percent)
    print("\n")
