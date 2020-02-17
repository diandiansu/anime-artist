
import cv2
from os import listdir
from os.path import isfile, join

# relative file path in local folder
img_folder_path = '../faces'
sketch_output_path = '../generated_sketch_data/'

# generate all sketchs from anime faces
for f in listdir(img_folder_path):
    if isfile(join(img_folder_path, f)):
        img = cv2.imread(join(img_folder_path, f), 1)
        # print(img)
        # converting the image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # inverting the image
        img_invert = cv2.bitwise_not(img_gray)
        # bluring or smoothing the inverted image with the kernel size (10,10)
        img_blur = cv2.blur(img_invert, (10, 10))

        """
        The Dodge blend function divides the bottom layer by the inverted top layer.
        This lightens the bottom layer depending on the value of the top layer.
        We have the blurred image, which highlights the boldest edges
        """
        final_img = cv2.divide(img_gray, 255-img_blur, scale=256)
        save_path = sketch_output_path + f.split(".")[0] + '.png'
        cv2.imwrite(save_path, final_img)


cv2.destroyAllWindows()
