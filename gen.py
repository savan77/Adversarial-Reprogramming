from PIL import Image, ImageDraw
from os import path as osp
import os
import numpy.random as rdn

class ImageGenerator:
	"""
		Class to generate adversaril images.
	"""
	def __init__(self, bgcolor=(0,0,0), output_dir="output/", num_classes=10, num_squares=16):
		self.bgcolor = bgcolor
		self.count = 0
		self.out_dir = output_dir
		self.num_classes = list(range(1,num_classes+1))
		self.num_squares = list(range(1,num_squares+1))
		self._test_out_dir()

	def save_output(self, img, label):
		img.save(osp.join(self.out_dir, str(self.count)+".jpg"))
		print("Saving {}".format(osp.join(self.out_dir, str(self.count)+".jpg")))
		with open(osp.join(self.out_dir, str(self.count)+".txt"), "w") as f:
			f.write(str(label))

	def get_prob(self):
		weights = [0.1] * 10
		label = int(rdn.choice(self.num_classes, 1, p=weights))
		where = rdn.choice(self.num_squares, label, replace=False)
		return label, where

	def draw_squares(self, img):
		label, where = self.get_prob()
		draw = ImageDraw.Draw(img)
		place_map = {1:[0,0,9,9],2:[18,0,27,9],3:[9,9,18,18],4:[27,9,36,18],5:[0,18,9,27],\
		6:[9,18,18,27],7:[27,0,36,9],8:[18,18,27,27],9:[18,27,27,36],10:[9,27,18,36],11:[9,0,18,9],\
		12:[18,9,27,18],13:[27,27,36,36],14:[0,9,9,18],15:[0,27,9,36],16:[27,18,36,27]}
		for i in range(label):
			place = where[i]
			grid = place_map[place]
			draw.rectangle(grid, fill=(255,255,255), outline=(0,0,0))

		return img, label

	def _test_out_dir(self):
		if not osp.exists(self.out_dir):
			os.mkdir(self.out_dir)

	def generate(self, num=10):
		for i in range(num):
			img = Image.new('RGB', (36,36), color=(0,0,0))
			#draw rectangle
			img, label = self.draw_squares(img)
			#save img
			self.save_output(img, label)
			self.count += 1

def main():
	
	# img = Image.new('RGB', (36,36), color=(0,0,0))
	# draw = ImageDraw.Draw(img)
	# draw.rectangle([0,0,9,9], fill=(255,255,255), outline=(0,0,0))
	# draw.rectangle([9,0,18,9], fill=(255,255,255), outline=(0,0,0))
	# draw.rectangle([18,0,27,9], fill=(255,255,255), outline=(0,0,0))
	# draw.rectangle([27,0,36,9], fill=(255,255,255), outline=(0,0,0))
	# img.save('test.jpg')

	generator = ImageGenerator()
	generator.generate()

if __name__ == "__main__":
	main()

