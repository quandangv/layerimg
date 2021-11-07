from PIL import Image, ImageOps, ImageDraw, ImageFont
import main
import collections
import os

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
  #main.action_test(root + '.png')
  with Image.open(root + '.jpg') as jpeg, Image.open(root + '_out.png') as our_output:
    result = Image.new('RGB', (size[0]*2, size[1]))
    result.paste(ImageOps.scale(jpeg.crop(crop_box), scale, Image.BOX))
    result.paste(ImageOps.scale(our_output.crop(crop_box), scale, Image.BOX), (size[0], 0))
    result = ImageOps.autocontrast(result, preserve_tone=True)

    jpeg_size = format_bytes(os.path.getsize(root + '.jpg'))
    our_size = format_bytes(os.path.getsize(root + '.zip'))
    font = ImageFont.truetype('boldfont.ttf', 80)
    draw = ImageDraw.Draw(result)
    draw.text((size[0]/2, size[1]-40), f"JPEG {jpeg_size}", font=font, anchor='ms', fill=color)
    draw.text((size[0]*3/2, size[1]-40), f"LAYERIMG {our_size}", font=font, anchor='ms', fill=color)
    draw.line([ size[0], 0, size[0], size[1] ], width=8, fill=color)
    result.save(path)

main.args = Args(True, 0.8, 0.8, 0.8, False)
make_comparison('output/kurzgesagt', (550, 40), 'demo/letter.png')
main.args = Args(True, 0.8, 0.7, 0.4, False)
make_comparison('output/landscape', (680, 80), 'demo/mountain.png', (0, 0, 0))
main.args = Args(True, 0.8, 0.8, 1, False)
make_comparison('output/vector', (70, 550), 'demo/sheep.png')
