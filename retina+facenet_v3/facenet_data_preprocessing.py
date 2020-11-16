A_P_N_DF = []
data_path = './data'
vein_path = [data_path + '/' + dir_path for dir_path in os.listdir(data_path)]

for p_vein_path in tqdm(vein_path):
    for n_vein_path in vein_path:
        if p_vein_path != n_vein_path: 
            p_vein_img_paths = [p_vein_path + '/' + img_path for img_path in os.listdir(p_vein_path)]
            n_vein_img_paths = [n_vein_path + '/' + img_path for img_path in os.listdir(n_vein_path)]
            for i in range(len(p_vein_img_paths)):
                a_vein_img_path = p_vein_img_paths[i]
                for j in range(i+1, len(p_vein_img_paths)):
                    p_vein_img_path = p_vein_img_paths[j]
                    for k in range(len(n_vein_img_paths)):
                        n_vein_img_path = n_vein_img_paths[k]
                        A_P_N_DF.append({'anchor' : a_vein_img_path,
                                         'positive' : p_vein_img_path, 
                                         'negative' : n_vein_img_path})
A_P_N_DF = pd.DataFrame(A_P_N_DF)

A_P_N_ARR = []

for i, instance in A_P_N_DF.iterrows():
    A_P_N = []
    anchor = cv2.imread(instance['A'])
    # cv2.resize(anchor, )
    positive = cv2.imread(instance['P'])
    negative = cv2.imread(instance['N'])

    A_P_N.append(anchor)
    A_P_N.append(positive)
    A_P_N.append(negative)
    
    A_P_N_ARR.append(A_P_N)

A_P_N_ARR = np.array(A_P_N_ARR)
print(f'result : {A_P_N_ARR.shape}')