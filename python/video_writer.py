import cv2

class video_writer:
	def __init__(self, out_file_name, fourcc, fps, window_size):
		if not isinstance(out_file_name, str):
			raise TypeError("Parametr out_file_name must be a sting.")
		self.out = cv2.VideoWriter(out_file_name, fourcc, fps, window_size)

	def write(self, img):
		self.out.write(img)

	def __del__(self):
		self.out.release()


