import numpy as np
import cv2 as cv

def bgr2rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def dilate_mask(mask, size = 20):
  kernel = np.ones((size,size), np.uint8)
  img_dilation = cv.dilate(mask.astype(np.uint8), kernel, iterations=1)
  return img_dilation.astype(bool)

def apply_masks_to_img(image, masks):
    merged_masks = merge_masks(masks)
    return apply_mask_to_img(image, merged_masks)

def merge_masks(masks):
    assert len(masks) > 0

    res_mask = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        res_mask[mask == True] = True

    return res_mask

def prep_image_for_inpainting(img, masks):
  res_img = apply_masks_to_img(img, masks)
  return bgr2rgb(res_img) / 255

def prep_mask_for_inpaiting(mask):
    scaled_mask = np.zeros((*mask.shape, 3), dtype=bool)
    scaled_mask[mask]=(True, True, True)

    return np.abs(1 - scaled_mask.astype(np.int8))

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

        processed_masks.append([obj_mask, obj_shadow_mask])

    return np.array(processed_masks)