
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from torchvision import io, transforms
import cv2
import yaml
import torch
from torch.nn import functional as F

from sys import exit as e


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



def show_animation(output_imgs, cap, deformed, configs):
  ims= []
  fig = plt.figure(figsize=(16, 8))
  # fig, axs = plt.subplots(1, 2, figsize=(16, 8))
  ax1 = fig.add_subplot(1, 3, 1)
  ax2 = fig.add_subplot(1, 3, 2)
  ax3 = fig.add_subplot(1, 3, 3)
  inter = configs["params"]["vid_interval"] if configs["params"]["type"]=="video"\
    else configs["params"]["img_interval"]
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

  for i in range(len(output_imgs)):
    im_opt = ax1.imshow(output_imgs[i], animated=True)
    im_real = ax2.imshow(cap[i], animated=True)
    im_deformed = ax3.imshow(deformed[i], animated = True)
    ims.append([im_real, im_opt, im_deformed])
    ani = animation.ArtistAnimation(fig, ims, interval=inter, blit=True, repeat_delay = 1000)
  if int(configs["params"]["save_flg"]):
    ani.save('./output/smiling_2.mp4', writer=writer)
  else:
    plt.show()


def read_frames(configs):
  input_file = configs["paths"]["input"]
  cap = None
  if configs["params"]["type"] == "video":
    cap = io.read_video(input_file, pts_unit = 'sec')[0].numpy()
  else:
    input_file2 = configs["paths"]["input2"]
    imsize = configs["params"]["img_size"]
    img1 = cv2.resize(cv2.imread(input_file)[:, :, :3], (imsize, imsize))
    img2 = cv2.resize(cv2.imread(input_file2)[:, :, :3], (imsize, imsize))
    cap = [img1, img2]
  return cap


def normalize_tensor(img_tsr):
  c, h, w = img_tsr.size()
  img_tsr = img_tsr.flatten()
  img_tsr -= img_tsr.min(0, keepdim=True)[0]
  img_tsr /= img_tsr.max(0, keepdim=True)[0]
  img_tsr = img_tsr.view(c, h, w)
  return img_tsr


def make_coordinate_grid(spatial_size, type):
  """
  Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
  """
  h, w = spatial_size
  xx = torch.arange(0, w).view(1 ,-1).repeat(h ,1)
  yy = torch.arange(0, h).view(-1 ,1).repeat(1 ,w)
  xx = xx.view(1 ,1 ,h ,w).repeat(1 ,1 ,1 ,1)
  yy = yy.view(1 ,1 ,h ,w).repeat(1 ,1 ,1 ,1)
  grid = torch.cat((xx ,yy) ,1).float()
  return grid



def flow_warping(flow, img):
  img_tsr = transforms.ToTensor()(img).unsqueeze(0)
  x, y, dim = flow.shape
  B, C, H, W = img_tsr.size()
  flow = torch.from_numpy(flow)
  flow = flow.permute(2, 0, 1).unsqueeze(0)
  grid = make_coordinate_grid([flow.shape[2], flow.shape[3]], flow.dtype)
  deformation = grid + flow
  deformation[: ,0 ,: ,:] = 2.0 *deformation[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
  deformation[: ,1 ,: ,:] = 2.0 *deformation[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0
  deformation = deformation.permute(0 ,2 ,3 ,1)
  flow = flow.permute(0, 2, 3, 1)
  warped = F.grid_sample(img_tsr, deformation)
  mask = torch.ones((img_tsr.size()))
  mask = F.grid_sample(mask, deformation)
  mask[mask<0.9999] = 0
  mask[mask>0] = 1
  output = warped * mask
  output = output.squeeze(0).permute(1, 2, 0)
  output = output.type(torch.float32)
  return output