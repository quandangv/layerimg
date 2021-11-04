from PIL import Image, ImageOps, ImageChops, ImageFilter
from math import *
from array import *
import zipfile
import sys
import operator
import subprocess
import tempfile

scale_exponent = 2
layer_count = 3
scale_method = Image.BOX
tmp_dir = 'output/'
debug = True
quality = 4
jpeg_loss = 50
fill_range = 8

def upscale(img):
  with tempfile.TemporaryDirectory() as tmp_dir:
    upscale_in = tmp_dir + '/upscale_in.png'
    upscale_out = tmp_dir + '/upscale_out.png'
    img.save(upscale_in)
    #subprocess.run(['./ersgan/realesrgan-ncnn-vulkan', '-n', 'realesrgan-x4plus-anime', '-i', upscale_in, '-o', upscale_out])
    subprocess.run(['./ersgan/realesrgan-ncnn-vulkan', '-i', upscale_in, '-o', upscale_out])
    return Image.open(upscale_out)

scale_ratio = 2 ** scale_exponent
scale = lambda img, n, method=scale_method: ImageOps.scale(img, n, method)
input_path = sys.argv[1]
output_path = sys.argv[2]

ranges = [[] for _ in range(fill_range+1) ]
for x in range(-fill_range, fill_range+1):
  for y in range(-fill_range, fill_range+1):
    dist = round(sqrt(x*x + y*y))
    if dist <= fill_range:
      ranges[dist].append((x,y))

def debug_save(img, name):
  if debug:
    img.save(tmp_dir + name)

# DECONSTRUCT
with Image.open(input_path) as og_img:
  # pad image to the correct size
  size_unit = scale_ratio ** layer_count
  my_round = lambda num: ceil(num/size_unit)*size_unit
  img = Image.new('RGB', (my_round(og_img.width), my_round(og_img.height)))
  img.paste(og_img)
  for x in range(og_img.width, img.width):
    for y in range(og_img.height):
      img.putpixel((x, y), og_img.getpixel((og_img.width-1, y)))
  for y in range(og_img.height, img.height):
    for x in range(og_img.width):
      img.putpixel((x, y), og_img.getpixel((x, og_img.height-1)))
  for x in range(og_img.width, img.width):
    for y in range(og_img.height, img.height):
      img.putpixel((x, y), og_img.getpixel((og_img.width-1, og_img.height-1)))
  og_size = og_img.size

debug_save(img, 'pad.png')

with zipfile.ZipFile(output_path, mode='w') as output:
  def save_layer(index, color, alpha):
    debug_save(scale(color, scale_ratio**index, Image.BOX), f'layer{index}.png')
    debug_save(scale(alpha.convert('L'), scale_ratio**index, Image.BOX), f'layer{index}_alpha.png')
    f = tempfile.NamedTemporaryFile()
    def save_image(img, f, quality):
      if quality > 95:
        img.save(f, 'PNG', optimize = True)
      else:
        img.save(f, 'JPEG', quality = quality, subsampling = 2, optimize = True)
    save_image(img, f, round(100 - jpeg_loss/scale_ratio**index))
    output.write(f.name, arcname=f'layer{index}')
    f.truncate(0)
    f.close()
    f = tempfile.NamedTemporaryFile()
    alpha.save(f, 'PNG', optimize = True)
    output.write(f.name, arcname=f'layer{index}_alpha')
    f.close()
  for layer_idx in range(layer_count):
    def calc_upscale_loss(og, downscaled):
      byte2log = lambda v: round(log(v+1)*46)
      with upscale(downscaled) as upscaled:
        diff = ImageChops.difference(og, upscaled).convert('L').convert('F')
      edges = og.filter(ImageFilter.FIND_EDGES).convert('L').convert('F')
      #edges = og.filter(ImageFilter.FIND_EDGES).convert('L').filter(ImageFilter.GaussianBlur(2)).convert('F')
      mean = 0
      for x in range(edges.width):
        for y in range(edges.height):
          mean += edges.getpixel((x, y))
      mean /= edges.width * edges.height
      print(mean)
      return diff.point(lambda a: a*(0.02))

    downscaled = scale(img, 1/scale_ratio)
    loss = calc_upscale_loss(img, downscaled)

    alpha = Image.new('1', img.size)
    #threshold = 1 / (quality * scale_ratio**(layer_idx*2))
    threshold = 1 / (quality * scale_ratio**layer_idx)
    #padded_width = img.width + fill_range*2
    #padded_height = img.height + fill_range*2
    #rank_map = bytearray(padded_width*padded_height)
    #color = Image.new('RGB', img.size)
    #for i in range(padded_width*padded_height):
    #  if i%padded_width < img.width and i//padded_width < img.height:
    #    rank_map[i] = fill_range
    for x in range(og_size[0]):
      for y in range(og_size[1]):
        xy = (x, y)
        if loss.getpixel(xy) > threshold:
          pix = img.getpixel(xy)
          alpha.putpixel(xy, 1)
          #for rank in range(fill_range):
          #  for d in ranges[rank]:
          #    dxy = (x+d[0], y+d[1])
          #    index = dxy[0]%padded_width + dxy[1]%padded_height*padded_width
          #    if rank_map[index] > rank:
          #      color.putpixel(dxy, pix)
          #      rank_map[index] = rank
    #save_layer(layer_idx, color, alpha)
    save_layer(layer_idx, img, alpha)
    debug_save(Image.merge('RGBA', (*img.split(), *alpha.convert('L').split())), f'layer{layer_idx}_combined.png')
    img = downscaled
    og_size = tuple(map(lambda a: ceil(a/scale_ratio), og_size))
  save_layer(layer_count, img, Image.new('1', img.size, 1))

# CONSTRUCT

with zipfile.ZipFile(output_path, mode='r') as archive, tempfile.TemporaryDirectory() as tmp_path:
  def load_layer(index):
    archive.extract(f'layer{index}', tmp_path)
    layer = Image.open(tmp_path + f'/layer{index}')
    archive.extract(f'layer{index}_alpha', tmp_path)
    alpha = Image.open(tmp_path + f'/layer{index}_alpha')
    return Image.merge('RGBA', (*layer.split(), *alpha.convert('L').split()))

  img = load_layer(layer_count)
  for layer_idx in reversed(range(layer_count)):
    mask = load_layer(layer_idx)
    debug_save(scale(img, scale_ratio**(layer_idx+1), Image.BOX), f'stage{layer_idx}.png')
    with upscale(img) as upscaled:
      img = Image.alpha_composite(upscaled, mask)
  img.save(tmp_dir + 'result.png')
