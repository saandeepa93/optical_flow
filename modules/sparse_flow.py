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
from torchvision import io, transforms
from sys import exit as e

from modules.util import imshow, show
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
  input_file = configs["paths"]["input"]
  cap = io.read_video(input_file, pts_unit = 'sec')[0].numpy()
  lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  # feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
  color = (0, 255, 0)

  first_frame = cap[0]
  prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
  # prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
  prev = pred_landmarks(first_frame)[:, np.newaxis, :].astype(np.float32)
  mask = np.zeros_like(first_frame)

  for frame in cap[1:]:
    # frame_2_orig = np.copy(frame_2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # keypoints = np.argwhere(segments==1)
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    good_old = prev[status == 1]
    good_new = next[status == 1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
      a, b = new.ravel()
      c, d = old.ravel()
      mask = cv2.line(mask, (a, b), (c, d), color, 2)
      frame = cv2.circle(frame, (a, b), 3, color, -1)
    output = cv2.add(frame, mask)
    prev_gray = gray.copy()
    prev = good_new.reshape(-1, 1, 2)
    imshow(output)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

