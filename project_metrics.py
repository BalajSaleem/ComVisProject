from PIL import Image, ImageChops, ImageEnhance

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
