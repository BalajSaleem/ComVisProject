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