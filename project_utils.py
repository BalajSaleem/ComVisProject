import numpy as np
import cv2 as cv

def bgr2rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def apply_masks_to_img(img, masks, square=False):
    res_img = img.copy()
    for mask in masks:
        res_img = apply_mask_to_img(res_img, mask)

    return res_img

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
    indiv_masks_instances, obj_shadow_masks_instances = outputs

    indiv_masks = indiv_masks_instances[0]['instances'].pred_masks
    indiv_mask_assoc = indiv_masks_instances[0]['instances'].pred_associations
    indiv_pred_classes = indiv_masks_instances[0]['instances'].pred_classes
    obj_shadow_masks = obj_shadow_masks_instances[0]['instances'].pred_masks
    obj_shadow_mask_assoc = obj_shadow_masks_instances[0]['instances'].pred_associations
    
    processed_masks = []

    for i, pred_class in enumerate(indiv_pred_classes):
        if pred_class != 0:
            continue
        
        obj_mask = np.array(indiv_masks[i])
        assoc = indiv_mask_assoc[i]
        obj_shadow_mask_ind = np.where(np.array(obj_shadow_mask_assoc) == assoc)[0]

        if len(obj_shadow_mask_ind) == 0:
            obj_shadow_mask = obj_mask.copy()
        else:
            obj_shadow_mask = obj_shadow_masks[obj_shadow_mask_ind[0]].squeeze()

        processed_masks.append((obj_mask, obj_shadow_mask))

    return processed_masks