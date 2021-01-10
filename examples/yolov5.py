#!/usr/bin/env python3
import io
import pickle
from extra.utils import fetch, my_unpickle, fake_torch_load

import sys
sys.path.append("./examples/yoloTemps")
# TODO : Update requirements.txt :  pandas seaborn cv2

class YOLOv5:
  def __init__(self, number=0):
    self.number = number
  
  def load_weights(self):
    fetch_url = None
    if (self.number == 0):
      fetch_url = 'https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt'
    else:
      raise Exception("no pretrained weights")
    
    dat = fetch(fetch_url)

    #print(dat)
    

    import zipfile
    fp = zipfile.ZipFile(io.BytesIO(dat))
    #fp.printdir()
    data = fp.read('archive/data.pkl')

    
    import torch
    b0 = torch.load(io.BytesIO(dat))

    print(b0)
    print("---------------")

    # yolo specific
    ret, out = my_unpickle(io.BytesIO(data))
    d = ret['model'].yaml
    print(d)
    print("---------------")

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
      tm = ret['model']._modules['model'][i]
      print(i, f, n, m, args, tm._modules.keys())
      # Focus, Conv, BottleneckCSP, SPP, Concat, Detect
      #for k,v in tm._modules.items():
      #  print("   ", k, v)
      if m in "Focus":
        conv = tm._modules['conv']
        print("   Focus > ", conv._modules)
      if m in "Conv":
        conv, bn = tm._modules['conv'], tm._modules['bn']
        print("   Conv  > ", conv)


if __name__ == "__main__":
  model = YOLOv5()
  model.load_weights()
  