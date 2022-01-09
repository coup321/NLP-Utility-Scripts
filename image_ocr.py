import PIL as PIL
import pytesseract as tess
import matplotlib.pyplot as plt
import numpy as np

class Img2Str:

    def __init__(self, img_path=None):
        self.img_path = img_path
        self.image = None

    def load_image(self, img_path):
        self.img_path = img_path
        self.image = PIL.Image.open(img_path)
        return self.image           

    def run_ocr(self):
        if not self.img_path:
            raise AttributeError('self.image is not defined, load an image first.')
        self.load_image(self.img_path)
        text = tess.image_to_string(self.image)
        print(text)
        return text

    def show_image(self, figsize=(8,8)):
        """
        To show a PIL image you can use Image.show() class method, but this
        opens the file with default OS program through generation of a temp file.
        In order to show the image in Jupyter notebooks, one strategy is to 
        convert the image to a numpy array, and then use matplotlib.imshow()
        """
        if not self.img_path:
            raise AttributeError('self.image is not defined, load an image first.')
        self.load_image(self.img_path)
        as_np = np.asarray(self.image)
        plt.figure(figsize=figsize)
        plt.imshow(as_np)
        plt.axis('off')
        plt.show()

    def __call__(self, img_path):
        self.load_image(img_path)
        return self.run_ocr()

ocr = Img2Str('./ocr_img_test.png')
ocr.run_ocr()

