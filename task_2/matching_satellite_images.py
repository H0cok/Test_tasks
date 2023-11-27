import os
import numpy as np

from itertools import product

import rasterio
from rasterio.plot import reshape_as_image

import cv2

import matplotlib.pyplot as plt


def cut_and_save_image(img_path, output_dir, piece_size=2000):
	"""
	Cut the raster image into smaller pieces and save them as JP2 files.

	:param raster_img: The large raster image to be cut.
	:param raster_meta: Metadata of the raster image.
	:param output_dir: Directory to save the cut images.
	:param piece_size: Size of the smaller pieces, default is 2000x2000.
	"""
	with rasterio.open(img_path, "r", driver='JP2OpenJPEG') as src:
		raster_img = src.read()
		raster_meta = src.meta
	raster_img = reshape_as_image(raster_img)
	# Create the output directory if it doesn't exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Get the dimensions of the raster image
	height, width, channels = raster_img.shape

	# Calculate the number of pieces in each dimension
	num_pieces_horizontal = width // piece_size
	num_pieces_vertical = height // piece_size

	# Cut and save each piece
	for i in range(num_pieces_vertical):
		for j in range(num_pieces_horizontal):
			top = i * piece_size
			left = j * piece_size
			bottom = min((i + 1) * piece_size, height)
			right = min((j + 1) * piece_size, width)
			# Extract the piece
			piece = raster_img[top:bottom, left:right]

			# Modify metadata for the piece
			piece_meta = raster_meta.copy()
			piece_meta['width'], piece_meta['height'] = right - left, bottom - top
			piece_meta['transform'] = rasterio.windows.transform(
				rasterio.windows.Window(left, top, right - left, bottom - top),
				raster_meta['transform']
			)
			piece_meta['count'] = channels

			# Save the piece
			piece_filename = os.path.join(output_dir, f'piece_{i}_{j}.jp2')
			with rasterio.open(piece_filename, 'w', **piece_meta) as dest:
				# Convert piece back to rasterio format and write
				for k in range(channels):
					dest.write(piece[:, :, k], k + 1)
	return raster_img


def preprocess_gaussian_blur(image_path):
	"""
	Applies Gaussian blur preprocessing to an image at the given path.

	This function reads an image using rasterio, converts it to grayscale,
	and then applies a Gaussian blur. It returns the preprocessed image
	in 8-bit unsigned integer format.

	Parameters:
	- image_path (str): Path to the image to be preprocessed.

	Returns:
	- tuple: A tuple containing the blurred image in uint8 format and the original image.
	"""
	with rasterio.open(image_path, "r", driver='JP2OpenJPEG') as src:
		raster_img1 = src.read()
		raster_meta = src.meta
	image = reshape_as_image(raster_img1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	appied_img = cv2.GaussianBlur(gray, (5, 5), 0)
	if appied_img.dtype != np.uint8:
		# Normalize the image to 0-255 and convert to 8-bit unsigned integer
		normalized_image = cv2.normalize(appied_img, None, 0, 255, cv2.NORM_MINMAX)
		image_8u = np.uint8(normalized_image)
	else:
		image_8u = appied_img
	return image_8u, image


def match_images(img_dir_1, img_dir_2):
	"""
	Matches features between two large images by dividing them into smaller chunks,
	applying Gaussian blur, and using SIFT for feature detection and matching.

	The function cuts the images into smaller pieces, applies Gaussian blur, detects and matches
	features using SIFT, and then sorts the matches based on distance. If sufficient good matches
	are found, it displays the matched features.

	Parameters:
	- img_dir_1 (str): Directory of the first large image.
	- img_dir_2 (str): Directory of the second large image.

	Note: The function assumes the images are of equal size and divides them into 2000x2000 pieces.
	"""
	global_matches = []
	global_keypoints1 = []
	global_keypoints2 = []

	# Cut image into smaller pieces
	PIECE_SIZE = 2000
	output_dir_1 = "data/tmp_1"
	output_dir_2 = "data/tmp_2"

	image_1 = cut_and_save_image(img_dir_1, output_dir_1, PIECE_SIZE)

	image_2 = cut_and_save_image(img_dir_2, output_dir_2, PIECE_SIZE)

	# iterate over every piece and find needed keypoints
	for i, j in product(range(len(image_1) // PIECE_SIZE), repeat=2):

		# Get pathes of picture pieces
		path_1 = output_dir_1 + f"/piece_{i}_{j}.jp2"
		path_2 = output_dir_2 + f"/piece_{i}_{j}.jp2"

		# Apply the preprocessing function
		raster_img1, _ = preprocess_gaussian_blur(path_1)
		raster_img2, _ = preprocess_gaussian_blur(path_2)

		# Perform feature detection and matching
		sift = cv2.SIFT_create()
		keypoints1, descriptors1 = sift.detectAndCompute(raster_img1, None)
		keypoints2, descriptors2 = sift.detectAndCompute(raster_img2, None)

		if descriptors1 is None or descriptors2 is None or descriptors1.size == 0 or descriptors2.size == 0:
			continue

		# Matcher and matching
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		matches = bf.match(descriptors1, descriptors2)

		# leaving only matches with high distance value
		good_matches = [m for m in matches if m.distance < 80]
		dx, dy = i * 2000, j * 2000  # displacement based on chunk position
		for match in good_matches:
			# Clone the keypoints and adjust their coordinates
			kp1 = cv2.KeyPoint(keypoints1[match.queryIdx].pt[0] + dx,
			                   keypoints1[match.queryIdx].pt[1] + dy,
			                   keypoints1[match.queryIdx].size)
			kp2 = cv2.KeyPoint(keypoints2[match.trainIdx].pt[0] + dx,
			                   keypoints2[match.trainIdx].pt[1] + dy,
			                   keypoints2[match.trainIdx].size)

			global_keypoints1.append(kp1)
			global_keypoints2.append(kp2)
			global_matches.append(cv2.DMatch(len(global_keypoints1) - 1, len(global_keypoints2) - 1, match.distance))

	# Getting the result
	global_matches.sort(key=lambda x: x.distance)
	if len(global_matches) < 50:
		print("Images are not matching")
	else:
		match_color = (0, 255, 0)  # Green color for matches
		single_point_color = (255, 0, 0)  # Blue color for points
		matches_thickness = 2
		matches_img = cv2.drawMatches(
			image_1, global_keypoints1,
			image_2, global_keypoints2,
			global_matches[:100],
			None,
			matchColor=match_color,
			singlePointColor=single_point_color,
			flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
			matchesThickness=matches_thickness
		)

		plt.figure(figsize=(20, 10))  # Increase figure size to make details clearer
		plt.imshow(matches_img)
		plt.show()