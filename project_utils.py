import numpy as np
import cv2 as cv

def bgr2rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def apply_mask_to_img(img, mask, square=False):
    new_img = img.copy()

    if square:
        mask = square_mask(mask)

    new_img[mask] = (255, 255, 255)

    return new_img

def square_mask(mask):
    h_sums = np.sum(mask.astype(np.int8), axis=1)
    v_sums = np.sum(mask.astype(np.int8), axis=0)

    h_indexes = np.where(h_sums > 0)[0]
    v_indexes = np.where(v_sums > 0)[0]

    box = [h_indexes[0], h_indexes[-1] + 1, v_indexes[0], v_indexes[-1] + 1]

    print(box)

    res_mask = mask.copy()
    res_mask[box[0]:box[1], box[2]:box[3]] = True

    return res_mask

def process_lisa_outputs(outputs):
    indiv_masks, obj_shadow_masks = outputs

    indiv_masks = indiv_masks[0]['instances'].pred_masks
    obj_shadow_masks = obj_shadow_masks[0]['instances'].pred_masks

    assert len(indiv_masks) == 2 * len(obj_shadow_masks)
    
    processed_masks = []
    for i, obj_shadow_mask in enumerate(obj_shadow_masks):
        processed_masks.append((obj_shadow_mask.squeeze(), np.array(indiv_masks[i])))

    return processed_masks