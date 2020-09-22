import dlib
import cv2
import numpy as np

def pred_landmarks(image):
  det_flag = 0 #0 is dlib and 1 is CNN
  hog_face_detector = dlib.get_frontal_face_detector()

  dets = hog_face_detector(image, 1)
  if len(dets) == 0:
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')
    dets = cnn_face_detector(image, 1)
    if len(dets) == 0:
      print("No face")
      return
    det_flag = 1
    for face in dets:
      x = face.rect.left()
      y = face.rect.top()
      w = face.rect.right() - x
      h = face.rect.bottom() - y

      # draw box over face
      cv2.rectangle(image, (x + 20,y + 20), (x+w+20,y+h+20), (0,0,255), 2)

  else:
    for face in dets:
      x = face.left()
      y = face.top()
      w = face.right() - x
      h = face.bottom() - y

      # draw box over face
      # print(f"left: {x}, top:  {y}, right: {face.right()}, bottom:\
        # {face.bottom()}")

  predictor = \
  dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
  if det_flag == 0:
    landmark = np.zeros((68, 2))
    # frontal = np.zeros((2,2))
    for d, shape in [(d, predictor(image, d)) for _,d in enumerate(dets)]:
      xmin, xmax, ymin, ymax = d.left(), d.right(), d.top(), d.bottom()
      # frontal = np.array([[xmin, ymin], [xmax, ymax]])
      cnt = 0
      for i in range(shape.num_parts):
        lx, ly = shape.part(i).x, shape.part(i).y
        landmark[cnt,:] = np.array([lx, ly])
        # cv2.circle(image, (shape.part(i).x, shape.part(i).y), 4, (255, 0, 0), 2)
        cnt+=1
    return landmark
  else:
    for shape in [predictor(image, d.rect) for _,d in enumerate(dets)]:
      cnt = 0
      for i in range(shape.num_parts):
        # cv2.circle(image, (shape.part(i).x, shape.part(i).y), 4, (0, 0, 255), 2)
        lx, ly = shape.part(i).x, shape.part(i).y
        landmark[cnt,:] = np.array([lx, ly])
        cnt+=1
    return landmarks

  # cv2.imshow("face detection with dlib", image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

