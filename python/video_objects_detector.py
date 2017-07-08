#!/usr/bin/python3

from video_writer import *
from utils import * 
from rails_finder import * 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import time
camera_id = 0 

video_cap = cv2.VideoCapture(input('Specify the path to the video: '))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = video_writer("output1.avi", fourcc, 15, (1280, 720))

#class train_finder:
#	def __init__(self, train_cascade):
#		self.train_cascade = train_cascade
#	def find(self, img):
#		return (x1, y1, x2, y2)

train_cascade = cv2.CascadeClassifier('cascade.xml')

cv2.namedWindow('IMAGE')

### MATRIX PARAMETRS
f = 0.001 # focus distantion
H = 3 # train height
pixel_size = 0.000001 # http://bootsector.livejournal.com/43436.html
minamal_distance = 200	# TODO

points = [] # points of rails from current and previous iterations
safety_distance = 0	# last safety distance
calculated_distance = 0 # safety distance
s = 0 # updated distantion TODO
speed = 3.5
downest_c = downest_d = (0, 0)
current_a = current_b = (0, 0)
current_y = 0

fps = 15 # get from video information

# data for diagram
start_time = 0 
current_time = 1 / fps 
all_time = 0

times = []
distances = []
real_distance = []


detecting_zone = {'x_from': 0, 'x_to': 0, 'y_from': 0, 'y_to': 0}
height = width = 0

font = cv2.FONT_HERSHEY_SIMPLEX

if video_cap.isOpened():
	_, img = video_cap.read()
	height, width, _ = img.shape

	detecting_zone['x_from'] = width * 3 // 8
	detecting_zone['x_to'] = width * 5 // 8
	detecting_zone['y_from'] = height * 2 // 8 
	detecting_zone['y_to'] = height * 6 // 8

print(height)

rails_finder = Rails_finder()

last_a = last_b = last_c = last_d = None
min_edge_real_length = max_edge_real_length = None

while(video_cap.isOpened()):
	all_time += 1 / fps
	_, img = video_cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	calculated_distance  -= speed * (1 / fps)
	current_time += 1/fps

	a, b, c, d = rails_finder.find_rails(img, detecting_zone, gray = gray)
	if a != None:
		if c[0] < a[0] < b[0] < d[0]:
			min_edge_real_length = f * 1.5 / ((b[0] - a[0]) * pixel_size)
			max_edge_real_length = f * 1.5 / ((d[0] - c[0]) * pixel_size)
	
			last_a = a
			last_b = b
			last_c = c
			last_d = d

			last_min_real_edge = min_edge_real_length
			last_max_real_edge = max_edge_real_length

			s = (min_edge_real_length ** 2 - H ** 2) ** 0.5;
			if s < 200:
				if  calculated_distance < s:
					safety_distance = s
					calculated_distance  = s

	if min_edge_real_length != None:
		if calculated_distance <= s:
			max_last_a = last_a
			max_last_b = last_b
			max_last_c = last_c
			max_last_d = last_d
			if downest_c[1] < max_last_c[1]:
				downest_c = max_last_c
				downest_d = max_last_d
			max_last_min_real_edge = last_min_real_edge
			max_last_max_real_edge = last_max_real_edge

		kx = last_a[0] - (height - last_a[1]) * ((last_a[0] - last_c[0]) / (last_c[1] - last_a[1]))
		lx = last_b[0] + (height - last_b[1]) * ((last_d[0] - last_b[0]) / (last_d[1] - last_b[1]))

		k = (kx, height)
		l = (lx, height)

		current_min_edge_real_length = (calculated_distance ** 2 + H ** 2) ** 0.5
		delta_min_edge_pixel_length = f * 1.5 / (current_min_edge_real_length * pixel_size) - f * 1.5 / (max_last_min_real_edge * pixel_size)
			
		current_y = max_last_a[1] - (last_c[1] - max_last_a[1]) * (delta_min_edge_pixel_length) // (max_last_max_real_edge - (max_last_d[0] - max_last_c[0]))

		current_a = ((max_last_a[0] - delta_min_edge_pixel_length // 2), current_y)
		current_b = ((max_last_b[0] + delta_min_edge_pixel_length // 2), current_y)
			
		if  calculated_distance > 20:
			cv2.fillConvexPoly(img, np.array([current_a, current_b, l, k], 'int32'), (0, 255, 0))
			cv2.line(img, last_a, last_b, (0,0,255), 3)
			cv2.line(img, last_c, last_d, (0,0,255), 3)
			cv2.putText(img, "%dm" % calculated_distance, (int(current_a[0] + (current_b[0] - current_a[0]) / 2 - 16), int(current_a[1] + (height - current_a[1]) // 2)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
		elif calculated_distance > 10:
			cv2.fillConvexPoly(img, np.array([current_a, current_b, l, k], 'int32'), (0, 0, 255))
			cv2.putText(img, "%dm" % calculated_distance, (int(current_a[0] + (current_b[0] - current_a[0]) / 2 - 16), int(current_a[1] + (height - current_a[1]) // 2)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


	trains = train_cascade.detectMultiScale(gray[detecting_zone.get('y_from') : detecting_zone.get('y_to'),\
											detecting_zone.get('x_from') : detecting_zone.get('x_to')],\
											1.3, 1)

	cv2.rectangle(img, (detecting_zone.get('x_from'), detecting_zone.get('y_from')), (detecting_zone.get('x_to'), detecting_zone.get('y_to')),(255,0,0),2)
	for (x,y,w,h) in trains:
		d = f * H / (h * pixel_size)
		if d < minamal_distance:
			minamal_distance = ceil(d)

		x += int(width * 3 / 8)
		y += int(height * 2 // 8)
		cv2.rectangle(img,(x, y),(x + w, y + h),(255,0,0),2)

	cv2.putText(img, 'Haar detected object: ' + str(minamal_distance) + "meters", (width - 300, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	if safety_distance > 20:
		cv2.putText(img, 'last safety distance: ' + str(s) + "meters", (width - 300, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	else:
		cv2.putText(img, 'last safety distance: ' + str(s) + "meters", (width - 300, 60), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	times.append(current_time)
	distances.append((calculated_distance , 20)[calculated_distance < 20])
	real_distance.append(times[-1] * speed)

	if  calculated_distance > 20:
		cv2.putText(img, 'Calculated safety distance: ' + str(calculated_distance) + "meters", (width - 300, 90), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	else:
		cv2.putText(img, 'Calculated safety distance: ' + str(calculated_distance) + "meters", (width - 300, 90), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
	cv2.putText(img, 'Speed: 3km/h (%f m/s)' % speed, (width - 300, 120), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

	writer.write(img)
#	out.write(img)

	cv2.imshow('IMAGE', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video_cap.release()

cv2.destroyAllWindows()

plt.plot(times, distances)
plt.plot(times, [i for i in map(lambda x: real_distance[-1] - x, real_distance)])
s = 0
prev_w = 0
for w, h in zip(times, distances):
	s += (w - prev_w) * h
	prev_w = w

print(all_time)

plt.plot(times, [20 for i in times]) 
plt.plot(times, [s / times[-1] for i in times])
plt.xlabel('time(s)')
plt.ylabel('distance(m)')
plt.axis([0, 100, 0, 200])
plt.show()

