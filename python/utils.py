import cv2
from collections import namedtuple

Point = namedtuple('Point', 'x y')
Rectangle = namedtuple('Rectangle', 'x_from, x_to, y_from, y_to')

def get_ROI(img, coords):
	"return roi. Coords must be dict with 'x_from', 'x_to', 'y_from', 'y_to' keys"
	try:
		return img[coords['y_from']: coords['y_to'], coords['x_from']: coords['x_to']]
	except KeyError:
		return None

