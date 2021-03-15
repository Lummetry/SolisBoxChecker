"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI
@project: 
@description:
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import itertools
import constants as ct

from libraries_pub import LummetryObject

LBL_PERS = [ct.LBL_PERSON, ct.LBL_PERSOANA]

NR_PERS = {
  '1.png'   : 3,
  '207.png' : 4,
  '532.png' : 3,
  '646.png' : 4,
  '748.png' : 4
  }

class VaporBoxCheck(LummetryObject):
  def __init__(self, log, **kwargs):
    super().__init__(log=log, **kwargs)    
    return
  
  def startup(self):
    self.log.platform()
    self.log.gpu_info()    
    self._cfg_inf = self.log.load_json('inference.txt', verbose=False)
    self._opencv_ok = False
    self._pytorch_ok = False
    self._tensorflow_ok = False
    return
  
  def _load_images(self):
    try:
      import cv2
      if self.DEBUG:
        self.log.p('Loading images')
      path_images = os.path.join(
        self.log.get_data_folder(), 
        self.config_data[ct.PATH_IMAGES]
        )
      lst_names = os.listdir(path_images)
      lst_paths = [os.path.join(path_images, x) for x in lst_names]
      lst_imgs = [cv2.imread(x) for x in lst_paths]
      self.dct_imgs = dict(zip(lst_names, lst_imgs))
      if self.DEBUG:
        self.log.p('{} images loaded'.format(len(self.dct_imgs)))
    except:
      self.log.p('Images could not be loaded', 'r')
    return
  
  def _run_pytorch(self):
    try:
      try:
        import torch as th
      except Exception as e:
        self.log.p('Pytorch not loaded. Please check if Pytorch is configured \
                   in the current environment')
        self.log.p('Pytorch tests cannot run on the current environment.')
        self.log.p('Exception: {}'.format(str(e)), color='r')
        return
      #end try-except
      
      try:
        import torchvision as tv
      except Exception as e:
        self.log.p('Torchvision not loaded. Please check if `torchvision` is configured \
                   in the current environment')
        self.log.p('Pytorch tests cannot run on the current environment.')
        self.log.p('Exception: {}'.format(str(e)), color='r')
        return
      #end try-except
      
      from inference import PytorchGraph
      self.log.p('Pytorch v{} working'.format(th.__version__))
      self.log.p('Torchvision v{} working'.format(tv.__version__))
      
      #loading graph    
      th_graph = PytorchGraph(
        log=self.log,
        config_graph=self._cfg_inf[ct.PYTORCH]
        )
      
      #infer
      lst_imgs = list(self.dct_imgs.values())
      self.log.p('Running inference ...')
      for i in range(2):
        dct_inf = th_graph.predict(lst_imgs)
      
      #check results
      lst_inf = dct_inf[ct.INFERENCES]
      lst_inf = list(itertools.chain.from_iterable(lst_inf))
      self.log.p(
        str_msg='{} / {} images classified'.format(len(self.dct_imgs), len(self.dct_imgs)),
        color='g'
        )
      
      total_time = self.log.get_timing('PYTORCHGRAPH_' + ct.TIMER_PREDICT)
      time_per_frame = total_time / len(self.dct_imgs)
      self.log.p(
        '{:.2f}s per frame, {:.2f}s per batch ({})'.format(
          time_per_frame,
          total_time,
          len(self.dct_imgs)
        ),
        color='g'
        )
      self._pytorch_ok = True
    except Exception as e:
      self.log.p(
        'Exception encountered in Pytorch step: {}'.format(str(e)), 
        color='r'
        )
    return
  
  def _run_tensorflow(self):
    try:
      try:
        import tensorflow as tf
      except Exception as e:
        self.log.p('Tensorflow not loaded. Please check if Tensorflow is \
                   configured in the current environment')
        self.log.p('Tensorflow tests cannot run on the current environment. Exiting.')
        self.log.p('Exception: {}'.format(str(e)), color='r')
        return
      #end try-except
      
      from inference import TensorflowGraph
      self.log.p('Tensorflow v{} working'.format(tf.__version__))
      
      #loading graph
      tf_graph = TensorflowGraph(
        log=self.log,
        config_graph=self._cfg_inf[ct.TENSORFLOW]
        )
      
      #infer
      lst_imgs = list(self.dct_imgs.values())
      self.log.p('Running inference ...')
      for i in range(2):
        dct_inf = tf_graph.predict(lst_imgs)
      
      #check results
      lst_inf = dct_inf[ct.INFERENCES]
      lst_inf = list(itertools.chain.from_iterable(lst_inf))
      lst_pers = list(filter(lambda x: x[ct.TYPE] in LBL_PERS, lst_inf))    
      self.log.p(
        str_msg='{} / {} persons detected'.format(
          len(lst_pers), sum(NR_PERS.values())
        ),
        color='g'
        )
      
      total_time = self.log.get_timing('TENSORFLOWGRAPH_' + ct.TIMER_PREDICT)
      time_per_frame = total_time / len(self.dct_imgs)
      self.log.p(
        str_msg='{:.2f}s per frame, {:.2f}s per batch ({})'.format(
          time_per_frame,
          total_time,
          len(self.dct_imgs)
          ),
        color='g'
        )
      self._tensorflow_ok = True
    except Exception as e:
      self.log.p(
        str_msg='Exception encountered in Tensorflow step: {}'.format(str(e)),
        color='r'
        )
    return
  
  def _check_opencv(self):
    try:
      import cv2
      self._opencv_ok = True
      self.log.p('OpenCV v{} working'.format(cv2.__version__))
    except:
      self.log.p('OpenCV not found.', color='r')
    return
  
  def run(self):
    self._check_opencv()
    if self._opencv_ok:
      self._load_images()
      self._run_pytorch()
      self._run_tensorflow()
    
    if all([self._opencv_ok, self._pytorch_ok, self._tensorflow_ok]):
      self.log.p(
        str_msg='Environment is properly functioning, please send ' + 
          'the following log file to Lummetry Team: {}'.format(self.log.log_file),
        color='g'
        )
    else:
      self.log.p(
        str_msg='Environment not properly functioning, please send ' +
          'the following log file to Lummetry Team: {}'.format(self.log.log_file),
        color='r'
        )
    #endif
    return
    
    
    
    
    
    
    
    
    