import cv2
import numpy as np
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
setup_logger()

from sys import exit as e

from modules.util import imshow, show, show_animation, read_frames, flow_warping
from modules.face_detector import pred_landmarks

# You will be implementing sparse optical flow in this file

def get_masks(im):
  cfg = get_cfg()
  cfg.MODEL.DEVICE='cpu'
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)
  # keypoints = np.asarray(outputs["instances"].pred_keypoints)
  masks = np.asarray(outputs["instances"].pred_masks)
  return masks[0].astype(np.uint8)


def calc_flow(configs):
  # input_file = "./input/01.mp4"
  output_imgs = []
  deformed = []
  cap = read_frames(configs)
  lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  # feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
  color = (0, 255, 0)

  first_frame = cap[0]
  prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
  # prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
  prev = pred_landmarks(first_frame)[:, np.newaxis, :].astype(np.float32)
  mask = np.zeros_like(first_frame)
  mask[..., 1] = 255
  output_imgs.append(first_frame)
  deformed.append(first_frame)

  for frame in cap[1:]:
    # frame_2_orig = np.copy(frame_2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # keypoints = np.argwhere(segments==1)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    deformed_src = flow_warping(flow, first_frame)
    deformed.append(deformed_src)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    output_imgs.append(rgb)
  if int(configs["params"]["animate"]):
    show_animation(output_imgs, cap, deformed, configs)


