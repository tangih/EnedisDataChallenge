def separate_x_res(N):
    is_valid = [False for i in range(N)]
    for m in range(12):
        if m%2 == 0:
            continue
        for i in range(m*48*30, (m+1)*48*30):
            is_valid[i] = True
    return is_valid