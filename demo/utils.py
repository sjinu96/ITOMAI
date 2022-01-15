import numpy as np

mask_mapping = {(124, 142, 138): 8, (255, 6, 0): 14, (125, 2, 0): 112, (223, 255, 0): 127,
 (127, 127, 127): 139, (75, 75, 216): 148, (106, 164, 80): 96, (25, 247, 255): 105, 
 (203, 177, 56): 110, (216, 109, 109): 118, (58, 68, 129): 119, (23, 170, 23): 123, 
 (117, 26, 35): 124, (195, 192, 116): 125, (95, 216, 125): 126, (0, 185, 47): 128, 
 (44, 236, 145): 134, (150, 81, 3): 135, (97, 129, 65): 141, (95, 98, 185): 147, 
 (92, 65, 20): 149, (230, 198, 145): 153, (54, 44, 247): 154, (62, 232, 245): 156,
 (208, 217, 226): 158, (26, 105, 26): 168, (74, 122, 129): 177, (75, 94, 34): 181,} # 기타

inverse_mapping=dict((v,k) for (k,v) in mask_mapping.items())



def convert_rgb_to_label(x, origin_label_map):
    # x : drawn by user
    # origin_label_map : label map by segmentation model
    # print(x.shape)
    # print(origin_label_map.shape)

    temp = np.zeros(origin_label_map.shape)
    for rgb in mask_mapping:
        # print((x==rgb).all(axis=2))
        temp[(x == rgb).all(axis = 2)] = mask_mapping[rgb]

    return temp.astype(int)

def convert_label_to_rgb(label_map):
    label_rgb=np.zeros(label_map.shape+(3,))
    for label in inverse_mapping:
        label_rgb[label_map==label]=inverse_mapping[label]
    
    last_mask = (label_rgb==0)
    label_rgb[(label_rgb==0).all(axis=2)] = inverse_mapping[134] # mountain

    assert (label_rgb==0).all(axis=2).sum()==0 # 라벨 0이면 안 됨.
    return label_rgb.astype(np.uint8)