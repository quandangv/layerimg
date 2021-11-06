from PIL import Image, ImageOps, ImageChops, ImageFilter, ImageStat
from math import *
import zipfile
import sys
import os
import operator
import subprocess
import tempfile
import argparse
import platform

######## COMMONS ########

scale_exponent = 2
layer_count = 3
scale_method = Image.BOX
alpha_bpp = 2
alpha_power = 2

scale_ratio = 2 ** scale_exponent
scale = lambda img, n, method=scale_method: ImageOps.scale(img, n, method)

debug_dir = 'output/'
def debug_save(img, name):
  if args.debug:
    img.save(debug_dir + name)

def upscale(img):
  with tempfile.TemporaryDirectory() as tmp_dir:
    upscale_in = tmp_dir + '/upscale_in.png'
    upscale_out = tmp_dir + '/upscale_out.png'
    img.save(upscale_in)
    platform_system = platform.system()
    if platform_system == 'Linux':
      executable = './ersgan/realesrgan-ncnn-vulkan-linux'
    elif platform_system == 'Windows':
      executable = 'ersgan\\realesrgan-ncnn-vulkan-windows.exe'
    elif platform_system == 'Darwin':
      executable = './ersgan/realesrgan-ncnn-vulkan-macos'
    else: raise OSError('Unsupported OS')
    subprocess.run([executable, '-n', 'realesrgan-x4plus-anime', '-i', upscale_in, '-o', upscale_out])
    #subprocess.run([executable, '-i', upscale_in, '-o', upscale_out])
    return Image.open(upscale_out)

######## COMPRESS ########

def compress(input_path, output_path):
  with Image.open(input_path) as og_img:
    # pad image to the correct size
    size_unit = scale_ratio ** layer_count
    unit_round = lambda num: ceil(num/size_unit)*size_unit
    img = Image.new('RGB', (unit_round(og_img.width), unit_round(og_img.height)))
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

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with zipfile.ZipFile(output_path, mode='w') as output:
    output.writestr('size', og_size[0].to_bytes(8, 'big') + og_size[1].to_bytes(8, 'big'))
    jpeg_loss = 100 - args.quality*100
    def save_layer(index, color, alpha):
      debug_save(scale(color, scale_ratio**index, Image.BOX), f'layer{index}.png')
      debug_save(scale(alpha.convert('L'), scale_ratio**index, Image.BOX), f'layer{index}_alpha.png')
      jpeg_quality = round(100 - jpeg_loss/scale_ratio**index)
      f = tempfile.NamedTemporaryFile()
      def save_image(img, f, jpeg_quality):
        if jpeg_quality > 95:
          img.save(f, 'PNG', optimize = True)
        else:
          img.save(f, 'JPEG', quality=jpeg_quality, subsampling=2, optimize=True)
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
            sequence[bit_idx].save(f, 'PNG', optimize=True)
            output.write(f.name, arcname=f'layer{index}_alpha{bit_idx}')
      #alpha.save(f, 'PNG', optimize=True)
      #output.write(f.name, arcname=f'layer{index}_alpha')
      save_1_sequence(alpha, alpha_bpp)
      f.close()
    for layer_idx in range(layer_count):
      def calc_upscale_loss(og, downscaled):
        with upscale(downscaled) as upscaled:
          diff = ImageChops.difference(og, upscaled)
        edges = diff.filter(ImageFilter.FIND_EDGES).convert('L')
        diff = diff.convert('L')
        diff_mean = 0
        edges_mean = 0
        for x in range(edges.width):
          for y in range(edges.height):
            edges_mean += edges.getpixel((x, y))
            diff_mean += diff.getpixel((x, y))
        return (diff.filter(ImageFilter.GaussianBlur(0.5)).convert('F'), 0.1*edges_mean/diff_mean)

      downscaled = scale(img, 1/scale_ratio)
      loss, loss_scalar = calc_upscale_loss(img, downscaled)

      alpha = Image.new('L', img.size)
      #threshold = 1 / (args.detail_level * scale_ratio**(layer_idx*2))
      threshold = 1 / (loss_scalar * args.detail_level * scale_ratio**layer_idx)
      for x in range(og_size[0]):
        for y in range(og_size[1]):
          xy = (x, y)
          alpha.putpixel(xy, round((loss.getpixel(xy) / threshold)**alpha_power *255))
      save_layer(layer_idx, img, alpha)
      debug_save(scale(Image.merge('RGBA', (*img.split(), *alpha.split())), scale_ratio**layer_idx, Image.BOX), f'layer{layer_idx}_combined.png')
      img = downscaled
      og_size = tuple(map(lambda a: ceil(a/scale_ratio), og_size))
    save_layer(layer_count, img, Image.new('L', img.size, 255))

######## EXTRACT ########

def extract(input_path, output_path):
  with zipfile.ZipFile(input_path, mode='r') as archive, tempfile.TemporaryDirectory() as tmp_path:
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
      #debug_save(scale(img, scale_ratio**(layer_idx+1), Image.BOX), f'stage{layer_idx}.png')
      with upscale(img) as upscaled:
        debug_save(scale(upscaled, scale_ratio**layer_idx, Image.BOX), f'stage{layer_idx}.png')
        img = Image.alpha_composite(upscaled, mask)
    size = archive.read('size')
    img = img.crop((0, 0, int.from_bytes(size[:8], 'big'), int.from_bytes(size[8:], 'big'))).save(output_path)

######## COMPARE ########

def compare(path1, path2):
  with Image.open(path1) as img1, Image.open(path2) as _img2:
    img2 = _img2.convert(img1.mode)
    diff = ImageChops.difference(img1, img2).convert('L')
    edge_diff = ImageChops.difference(img1.filter(ImageFilter.FIND_EDGES), img2.filter(ImageFilter.FIND_EDGES)).convert('L')
    print('COMPARISON RESULT')
    print('Difference: ', ImageStat.Stat(diff).mean[0])
    print('Edge Difference: ', ImageStat.Stat(edge_diff).mean[0])

######## ARG PARSER ########

parser = argparse.ArgumentParser(description="Image archiver using an AI upscale algorithm.")
parser.add_argument('--debug', action='store_true', help="Saves intermediary results for debug purpose")
subparsers = parser.add_subparsers(title='actions', description="Specifies an action to do", required=True, dest='action')

compress_args = argparse.ArgumentParser(add_help=False)
compress_args.add_argument('--detail-level', '-d', type=float, default=1, help="The level of detail in the image archive. Must be positive, default is 1")
compress_args.add_argument('--quality', '-q', type=float, default=0.7, help="The compression quality (from 0 to 1) of the image archive")

simple_io = argparse.ArgumentParser(add_help=False)
simple_io.add_argument('paths', metavar='input', action='append', help="Path to input file")
simple_io.add_argument('paths', metavar='output', action='append', nargs='?', help='Path to save the output file. Defaults to saving next to the input')

parser_batch = subparsers.add_parser('batch', help="Processes a list of inputs", aliases=['b'], parents=[compress_args])
parser_compress = subparsers.add_parser('compress', help="Compresses an image", aliases=['c'], parents=[simple_io, compress_args])
parser_extract = subparsers.add_parser('extract', help="Extracts image from an archive", aliases=['e'], parents=[simple_io])
parser_compare = subparsers.add_parser('compare', help="Compare 2 images on various metrics")
parser_test = subparsers.add_parser('test', help="Compresses an image and then extracts to compare the result", aliases=['t'], parents=[compress_args])

parser_compress.set_defaults(action='compress')
parser_extract.set_defaults(action='extract')
parser_compare.set_defaults(action='compare')
parser_test.set_defaults(action='test')

parser_test.add_argument('paths', metavar='input', action='append', help="Path to input file")
parser_test.add_argument('paths', metavar='archive_output', action='append', nargs='?', help="Path to save the output archive. Defaults to saving next to the input")
parser_test.add_argument('paths', metavar='image_output', action='append', nargs='?', help="Path to save the output image. Defaults to saving next to the output archive")

parser_compare.add_argument('paths', metavar='image1', action='append', help="The first image to compare")
parser_compare.add_argument('paths', metavar='image2', action='append', help='The second image to compare')

parser_batch.add_argument('action', choices=['compress', 'extract', 'test'], help="Action to perform on the inputs")
parser_batch.add_argument('batch', metavar='inputs', nargs='+', help="The list of inputs to process")

args = parser.parse_args(sys.argv[1:])
#print(args)
if hasattr(args, 'detail_level'):
  assert args.detail_level > 0, f"Invalid detail_level: {args.detail_level}"
if hasattr(args, 'quality'):
  assert args.quality > 0 and args.quality <= 1, f"Invalid quality: {args.quality}"

######## MAIN ########

def replace_ext(path, new_ext):
  return os.path.splitext(path)[0] + new_ext

def action_compress(input, output=None):
  compress(input, output or replace_ext(input, '.zip'))

def action_extract(input, output=None):
  extract(input, output or replace_ext(input, '.png'))

def action_compare(path1, path2):
  compare(path1, path2)

def action_test(input, archive_output=None, image_output=None):
  archive_output = archive_output or replace_ext(input, '.zip')
  if not image_output:
    image_output = image_output or replace_ext(archive_output, '.png')
    if image_output == input:
      image_output = replace_ext(image_output, '_out.png')
  compress(input, archive_output)
  extract(archive_output, image_output)
  print()
  compare(input, image_output)

if hasattr(args, 'batch'):
  for input in args.batch:
    globals()['action_' + args.action](input)
else:
  globals()['action_' + args.action](*args.paths)
