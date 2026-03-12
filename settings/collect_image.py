from PIL import ImageFont

font_path = "../fonts/Roboto-Regular.ttf"  # Route to custom source
font_size = 28  # Font size
DATA_DIR_ESTATIC = "./data/dataset_estatico"  # Folder for storing images
DATA_VIDEO_DIR = "./data/data_dinamico"  # Folder for storing videos
dataset_size = 2000  # Number of images per class
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # letters of the alphabet


def load_font():
    try:
        return ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Custom font not found. Using default font.")
        return None
