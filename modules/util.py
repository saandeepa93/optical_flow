
import matplotlib.pyplot as plt
import cv2
import yaml


def imshow(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def show(img):
  plt.imshow(img)
  plt.show()


def get_config(config_path):
  with open(config_path) as file:
    configs = yaml.load(file, Loader = yaml.FullLoader)
  return configs

