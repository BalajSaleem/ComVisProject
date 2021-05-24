from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def ela(im):
    im = Image.fromarray(im)

    resaved = 'image' + '.resaved.jpg'
    ela = 'image' + '.ela.png'

    im.save(resaved, 'JPEG', quality=95)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return max_diff

def alt_ela(im):
    im = Image.fromarray(im)

    resaved = 'image' + '.resaved.jpg'
    ela = 'image' + '.ela.png'

    im.save(resaved, 'JPEG', quality=95)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    return np.array(ela_im)

def rel_alt_ela(im, o_im):
    im_alt_ela = alt_ela(im)
    o_im_alt_ela = alt_ela(o_im)

    return round(np.sum(im_alt_ela - o_im_alt_ela) / im.size, 2)

def eval(results, ela_th = 1, alt_ela_th = 0.1):
    results = np.array(results)
    ela_res = results[:, 0] - results[:, 1]
    ela_obj_better = np.sum(ela_res > ela_th)
    ela_obj_shadow_better = np.sum(ela_res < -ela_th)
    ela_same = len(results) - ela_obj_better - ela_obj_shadow_better

    alt_ela_res = results[:, 2] - results[:, 3]
    alt_ela_obj_better = np.sum(alt_ela_res > alt_ela_th)
    alt_ela_obj_shadow_better = np.sum(alt_ela_res < -alt_ela_th)
    alt_ela_same = len(results) - alt_ela_obj_better - alt_ela_obj_shadow_better

    ela_obj_avg = round(np.mean(results[:, 0]), 2)
    ela_obj_shadow_avg = round(np.mean(results[:, 1]), 2)
    alt_ela_obj_avg = round(np.mean(results[:, 2]), 2)
    alt_ela_obj_shadow_avg = round(np.mean(results[:, 3]), 2)

    return {'ela': {'obj_better': ela_obj_better, 'obj_shadow_better': ela_obj_shadow_better, 'same': ela_same, 
                    'obj_avg': ela_obj_avg, 'obj_shadow_avg': ela_obj_shadow_avg},
            'alt_ela': {'obj_better': alt_ela_obj_better, 'obj_shadow_better': alt_ela_obj_shadow_better, 'same': alt_ela_same,
                    'obj_avg': alt_ela_obj_avg, 'obj_shadow_avg': alt_ela_obj_shadow_avg}}