import dlib
from PIL import Image
import argparse

from imutils import face_utils
import numpy as np

import moviepy.editor as mpy

parser = argparse.ArgumentParser()
parser.add_argument("-image", required=True,help="path to input image")
args=parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68.dat')

max_width= 500

img=Image.open(args.image).convert('RGBA')

deal=Image.open("deals.png")
text=Image.open("text.png")

if img.size[0]>max_width:
	scaled_height = int(max_width*img.size[1]/img.size[0])
	img.thumbnail((max_width,scaled_height))

img_gray = np.array(img.convert('L'))

rects = detector(img_gray,0)

if len(rects) == 0:
	print("No images found")
	exit()

print("%i faces found"%len(rects)) 

faces=[]

for rect in rects:
	face={}
	print(rect.top(),rect.right(),rect.bottom(),rect.left())
	shades_width=rect.right()-rect.left()

	shape = predictor(img_gray,rect)
	shape = face_utils.shape_to_np(shape)

	leftEye = shape[36:42]
	rightEye = shape[42:48]

	leftEyeCenter = leftEye.mean(axis=0).astype("int")
	rightEyeCenter = rightEye.mean(axis=0).astype("int")

	dY = leftEyeCenter[1] - rightEyeCenter[1]
	dX = leftEyeCenter[0] - rightEyeCenter[0]
	angle=np.rad2deg(np.arctan2(dY,dX))

	current_deal = deal.resize((shades_width, int(shades_width * deal.size[1] / deal.size[0])),
                               resample=Image.LANCZOS)

	current_deal = current_deal.rotate(angle,expand=True)
	current_deal = current_deal.transpose(Image.FLIP_TOP_BOTTOM)

	face['glasses_image'] = current_deal
	left_eye_x = leftEye[0,0] - shades_width // 4
	left_eye_y = leftEye[0,1] - shades_width // 6
	face['final_pos'] = (left_eye_x, left_eye_y)
	faces.append(face)

duration = 4

def make_frame(t):
	draw_img=img.convert('RGBA')
	if t == 0:
		return np.asarray(draw_img)
	for face in faces:
		if t<= duration-2:
			current_x = int(face['final_pos'][0])
			current_y = int(face['final_pos'][1]*t/(duration-2))
			draw_img.paste(face['glasses_image'],(current_x,current_y),face['glasses_image'])

		else:
			draw_img.paste(face['glasses_image'],face['final_pos'],face['glasses_image'])
			draw_img.paste(text, (75, draw_img.height // 2 - 32), text)

	return np.asarray(draw_img)

animation = mpy.VideoClip(make_frame, duration=duration)
animation.write_gif("deal.gif", fps=4)

