#!/usr/bin/env python3
import io
import pickle
from extra.utils import fetch, my_unpickle, fake_torch_load

import sys
sys.path.append("./examples/yoloTemps")
# TODO : Update requirements.txt :  pandas seaborn cv2

USE_TORCH = True

class YOLOv5:
  def __init__(self, number=0):
    self.number = number
  
  def forward(self, x):
    # TODO : return outputs
    pass

  def load_weights(self):
    # TODO : load model properly
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

    if USE_TORCH:
      import torch
      b0 = torch.load(io.BytesIO(dat))
    else:
      b0 = fake_torch_load(dat)
    
    print(b0)
    print("---------------")
    return
    for k,v in b0.items():
      if '_blocks.' in k:
        k = "%s[%s].%s" % tuple(k.split(".", 2))
      mk = "self."+k
      #print(k, v.shape)
      try:
        mv = eval(mk)
      except AttributeError:
        try:
          mv = eval(mk.replace(".weight", ""))
        except AttributeError:
          mv = eval(mk.replace(".bias", "_bias"))
      vnp = v.numpy().astype(np.float32) if USE_TORCH else v
      vnp = vnp if k != '_fc.weight' else vnp.T

      if mv.shape == vnp.shape or vnp.shape == ():
        mv.data[:] = vnp
      else:
        print("MISMATCH SHAPE IN %s, %r %r" % (k, mv.shape, vnp.shape))

    return
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
  