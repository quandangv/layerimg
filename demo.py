from PIL import Image, ImageOps, ImageDraw, ImageFont
import main
import collections
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Generate demo images for the algorithm.")
parser.add_argument('--keep', action='store_true', help="Use the previous compression result instead of re-executing the algorithm")
args = parser.parse_args(sys.argv[1:])

Args = collections.namedtuple('Args', [ 'create_jpeg', 'quality', 'quality_gain', 'detail_level', 'debug' ])

def format_bytes(size):
  power = 2**10
  n = 0
  power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
  while size > power:
    size /= power
    n += 1
  return '%.1f' % size + power_labels[n]

def make_comparison(root, pos, path, color=(255, 255, 255), size=(100, 100), scale=8):
  crop_box = (*pos, pos[0] + size[0], pos[1] + size[1])
  size = (size[0]*scale, size[1]*scale)
  if not args.keep:
    main.action_test(f'input/{root}.png', f'output/{root}.zip')
  with Image.open(f'output/{root}.jpg') as jpeg, Image.open(f'output/{root}.png') as our_output:
    result = Image.new('RGB', (size[0]*2, size[1]))
    result.paste(ImageOps.scale(jpeg.crop(crop_box), scale, Image.BOX))
    result.paste(ImageOps.scale(our_output.crop(crop_box), scale, Image.BOX), (size[0], 0))
    result = ImageOps.autocontrast(result, preserve_tone=True)

    jpeg_size = format_bytes(os.path.getsize(f'output/{root}.jpg'))
    our_size = format_bytes(os.path.getsize(f'output/{root}.zip'))
    font = ImageFont.truetype('boldfont.ttf', 80)
    draw = ImageDraw.Draw(result)
    draw.text((size[0]/2, size[1]-40), f"JPEG {jpeg_size}", font=font, anchor='ms', fill=color)
    draw.text((size[0]*3/2, size[1]-40), f"LAYERIMG {our_size}", font=font, anchor='ms', fill=color)
    draw.line([ size[0], 0, size[0], size[1] ], width=8, fill=(255, 255, 255))
    result.save(path)
    return result

demo = [ None, None, None, None ]
main.args = Args(True, 0.8, 0.6, 0.6, False)
demo[0] = make_comparison('vector-noise', (350, 150), 'demo/circles.png', size=(200, 200), scale=4)
main.args = Args(True, 0.8, 0.8, 0.8, False)
demo[1] = make_comparison('kurzgesagt', (550, 40), 'demo/letter.png')
main.args = Args(True, 0.8, 0.7, 0.4, False)
demo[2] = make_comparison('landscape', (680, 80), 'demo/mountain.png', (0, 0, 0))
main.args = Args(True, 0.8, 0.8, 1, False)
demo[3] = make_comparison('vector', (70, 550), 'demo/sheep.png')

spacing = 100
demo = [ ImageOps.expand(img, spacing//2, (255, 255, 255)) for img in demo ]
# Assume all demo images have the same size
tiling = Image.new('RGB', (demo[0].width*2, demo[0].height*2))
tiling.paste(demo[0])
tiling.paste(demo[1], (demo[0].width, 0))
tiling.paste(demo[2], (0, demo[0].height))
tiling.paste(demo[3], (demo[0].width, demo[0].height))
tiling = ImageOps.expand(tiling, spacing//2, (255, 255, 255))
tiling.save('demo/jpeg-comparison.png')
