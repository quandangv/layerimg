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
debug = True
detail_level = 1
jpeg_loss = 50
fill_range = 8
alpha_bpp = 2
alpha_power = 2

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

tmp_dir = 'output/'
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
    jpeg_quality = round(100 - jpeg_loss/scale_ratio**index)
    f = tempfile.NamedTemporaryFile()
    def save_image(img, f, jpeg_quality):
      if jpeg_quality > 95:
        img.save(f, 'PNG', optimize = True)
      else:
        img.save(f, 'JPEG', quality = jpeg_quality, subsampling = 2, optimize = True)
    save_image(img, f, jpeg_quality)
    output.write(f.name, arcname=f'layer{index}')
    f.close()
    def save_1_sequence(img, bits_per_pixel):
      sequence = [ Image.new('1', img.size) for _ in range(bits_per_pixel) ]
      max = 2**bits_per_pixel-1
      for x in range(img.width):
        for y in range(img.height):
          xy = (x, y)
          val = round(img.getpixel(xy)/255*max)
          for idx in range(bits_per_pixel):
            sequence[idx].putpixel(xy, val%2)
            val //= 2
      for bit_idx in range(bits_per_pixel):
        with tempfile.NamedTemporaryFile() as f:
          sequence[bit_idx].save(f, 'PNG', optimize = True)
          output.write(f.name, arcname=f'layer{index}_alpha{bit_idx}')
    #alpha.save(f, 'PNG', optimize = True)
    #output.write(f.name, arcname=f'layer{index}_alpha')
    save_1_sequence(alpha, alpha_bpp)
    f.close()
  for layer_idx in range(layer_count):
    def calc_upscale_loss(og, downscaled):
      byte2log = lambda v: round(log(v+1)*46)
      with upscale(downscaled) as upscaled:
        diff = ImageChops.difference(og, upscaled).convert('L').convert('F')
      #edges = og.filter(ImageFilter.FIND_EDGES).convert('L').convert('F')
      #mean = 0
      #for x in range(edges.width):
      #  for y in range(edges.height):
      #    mean += edges.getpixel((x, y))
      #mean /= edges.width * edges.height
      #print(mean)
      return diff.point(lambda a: a*0.08)

    downscaled = scale(img, 1/scale_ratio)
    loss = calc_upscale_loss(img, downscaled)

    alpha = Image.new('L', img.size)
    threshold = 1 / (detail_level * scale_ratio**layer_idx)
    for x in range(og_size[0]):
      for y in range(og_size[1]):
        xy = (x, y)
        alpha.putpixel(xy, round((loss.getpixel(xy) / threshold)**alpha_power *255))
    save_layer(layer_idx, img, alpha)
    debug_save(Image.merge('RGBA', (*img.split(), *alpha.split())), f'layer{layer_idx}_combined.png')
    img = downscaled
    og_size = tuple(map(lambda a: ceil(a/scale_ratio), og_size))
  save_layer(layer_count, img, Image.new('L', img.size, 255))

# CONSTRUCT

with zipfile.ZipFile(output_path, mode='r') as archive, tempfile.TemporaryDirectory() as tmp_path:
  def load_layer(index):
    archive.extractall(tmp_path)
    color = Image.open(tmp_path + f'/layer{index}')
    def load_1_sequence(bits_per_pixel):
      sequence = [ Image.open(tmp_path + f'/layer{index}_alpha{bit_idx}') for bit_idx in range(bits_per_pixel) ]
      result = Image.new('L', sequence[0].size)
      max = 2**bits_per_pixel-1
      for x in range(result.width):
        for y in range(result.height):
          xy = (x, y)
          val = 0
          for idx in range(bits_per_pixel):
            if sequence[idx].getpixel(xy):
              val += 2**idx
          result.putpixel(xy, round(val/max*255))
      return result
    #alpha = Image.open(tmp_path + f'/layer{index}_alpha')
    alpha = load_1_sequence(alpha_bpp)
    debug_save(alpha, f'extracted_alpha{index}.png')
    return Image.merge('RGBA', (*color.split(), *alpha.split()))

  img = load_layer(layer_count)
  for layer_idx in reversed(range(layer_count)):
    mask = load_layer(layer_idx)
    debug_save(scale(img, scale_ratio**(layer_idx+1), Image.BOX), f'stage{layer_idx}.png')
    with upscale(img) as upscaled:
      img = Image.alpha_composite(upscaled, mask)
  img.save(tmp_dir + 'result.png')
