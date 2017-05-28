import cv2
import numpy as np

from utils import Point

class Rails_finder:
	def __init__(self):
		self.__points = []
		self.__prev_points = []
		pass

	def find_rails(self, img, detecting_zone, gray = None):
		gray = (gray, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[gray == None]
		edges = cv2.Canny(gray[:, detecting_zone.get('x_from') : detecting_zone.get('x_to')],50,120,apertureSize = 3)
		lines = cv2.HoughLinesP(edges,9,np.pi/180, 20, minLineLength=180, maxLineGap=10)
	
		if lines != None:
			lines = filter(lambda item: 0.1 < abs((item[0, 2] - item[0, 0]) / (item[0, 3] - item[0, 1])) < 0.25, lines)
			self.__prev_points = self.__points
			self.__points = []
			for line in lines:
				x1,y1,x2,y2 = line[0]
				self.__points += [(x1, y1, 1, (x2 - x1) / (y2 - y1)), (x2, y2, -1, (x2 - x1) / (y2 - y1))]
				(x1, x2) = (x1 + detecting_zone.get('x_from'), x2 + detecting_zone.get('x_from'))
				cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
			for x1, x2, y1, y2 in self.__prev_points:
				(x1, x2) = (x1 + detecting_zone.get('x_from'), x2 + detecting_zone.get('x_from'))
				cv2.line(img,(int(x1), int(y1)),(int(x2), int(y2)),(0,255,0),3)
	
			right_arrows = sorted(filter(lambda item: item[3] > 0, self.__points + self.__prev_points), key = lambda point: point[1])
			left_arrows = sorted(filter(lambda item: item[3] < 0, self.__points + self.__prev_points), key = lambda point: point[1])
			if len(left_arrows) > 0 and len(right_arrows) > 0:
				min_left = left_arrows[0]
				min_left_y = min_left[1]
				max_left = left_arrows[-1]
				max_left_y = max_left[1]
	
				min_right = right_arrows[0]
				min_right_y = min_right[1]
				max_right = right_arrows[-1]
				max_right_y = max_right[1]
	
				min_edge_y = max(min_left_y, min_right_y)
				max_edge_y = min(max_left_y, max_right_y)
				min_edge_left_x = int((min_left[0] - max_left[0]) * (min_edge_y - max_left[1]) / (min_left[1] - max_left[1]) + max_left[0]) #(Bx - Ax)*(Cy - Ay)/(By - Ay) + Ax <==> B = min_left A = max_left Cy = min_edge_y
				min_edge_right_x = int((min_right[0] - max_right[0]) * (min_edge_y - max_right[1]) / (min_right[1] - max_right[1]) + max_right[0]) #(Bx - Ax)*(Cy - Ay)/(By - Ay) + Ax <==> B = min_right A = max_right Cy = min_edge_y
				max_edge_left_x =  int((min_left[0] - max_left[0]) * (max_edge_y - max_left[1]) / (min_left[1] - max_left[1]) + max_left[0]) #(Bx - Ax)*(Cy - Ay)/(By - Ay) + Ax <==> B = min_left A = max_left Cy = min_edge_y
				max_edge_right_x = int((min_right[0] - max_right[0]) * (max_edge_y - max_right[1]) / (min_right[1] - max_right[1]) + max_right[0]) #(Bx - Ax)*(Cy - Ay)/(By - Ay) + Ax <==> B = min_right A = max_right Cy = min_edge_y
	
			
				return  (detecting_zone['x_from'] + min_edge_left_x, min_edge_y), (detecting_zone['x_from'] + min_edge_right_x, min_edge_y),\
						(detecting_zone['x_from'] + max_edge_left_x, max_edge_y), (detecting_zone['x_from'] + max_edge_right_x, max_edge_y)
		return None, None, None, None


