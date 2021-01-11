#!/usr/bin/env python3
import io
import pickle
import sys
import time
import cv2
import numpy as np
import os
from extra.utils import fetch, my_unpickle, fake_torch_load
from extra.yolov5 import YOLOv5
from tinygrad.tensor import Tensor

GPU = os.getenv("GPU", None) is not None

def infer(model, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # if you want to look at the image
  """
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()
  """

  # low level preprocess
  img = np.moveaxis(img, [2,0,1], [0,1,2])
  img = img.astype(np.float32)[:3].reshape(1,3,224,224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

  # run the net
  if GPU:
    out = model.forward(Tensor(img).gpu()).cpu()
  else:
    out = model.forward(Tensor(img))

  # if you want to look at the outputs
  """
  import matplotlib.pyplot as plt
  plt.plot(out.data[0])
  plt.show()
  """
  return out, retimg

if __name__ == "__main__":
  model = YOLOv5()
  model.load_weights()
  
  from PIL import Image
  url = sys.argv[1]
  if url == 'webcam':
    import cv2
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
      _ = cap.grab() # discard one frame to circumvent capture buffering
      ret, frame = cap.read()
      img = Image.fromarray(frame[:, :, [2,1,0]])
      out, retimg = infer(model, img)
      # print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
      SCALE = 3
      simg = cv2.resize(retimg, (224*SCALE, 224*SCALE))
      retimg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
      cv2.imshow('capture', retimg)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()
  else:
    if url.startswith('http'):
      img = Image.open(io.BytesIO(fetch(url)))
    else:
      img = Image.open(url)
    st = time.time()
    out, _ = infer(model, img)
    # print(np.argmax(out.data), np.max(out.data), lbls[np.argmax(out.data)])
    print("did inference in %.2f s" % (time.time()-st))