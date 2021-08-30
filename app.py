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

import traceback
import itertools
import pandas as pd
import constants as ct

from app_monitor import ApplicationMonitor
from libraries_pub import LummetryObject


LBL_PERS = [ct.LBL_PERSON, ct.LBL_PERSOANA]

CLF_OBJECTS = ['apple', 'chair', 'mountain_bike', 'soccer_ball', 'tabby']

NR_PERS = {
  '1.png'   : 3,
  '207.png' : 4,
  '532.png' : 3,
  '646.png' : 4,
  '748.png' : 4
  }
NR_RUN_INF = 3

__VER__ = '1.0.2'

class VaporBoxCheck(LummetryObject):
  def __init__(self, log, **kwargs):
    self.__version__ = __VER__
    super().__init__(log=log, **kwargs)    
    return
  
  def startup(self):    
    self._cfg_inf = self.log.load_json('inference.txt', verbose=False)
    self._opencv_ok = False
    self._pytorch_ok = False
    self._tensorflow_ok = False
    self._results = {
      'SCRIPT_VER': [self.__version__],
      'SYS_MEMORY': [],
      'SYS_PLATFORM': [],
      'GPU_NAME': [],
      'GPU_MEMORY': [],
      'OPENCV_VER': [],
      'TH_VER': [],
      'TF_VER': [],
      'TH_GPU_TIME': [],
      'TH_CPU_TIME': [],
      'TF_GPU_TIME': [],
      'TF_CPU_TIME': []
      }
    system, release, version = self.log.platform()    
    lst_gpus = self.log.gpu_info()    
    sys_memory = self.log.get_machine_memory()
    self.app_monitor = ApplicationMonitor(log=self.log)
    self._results['SYS_MEMORY'].append(sys_memory)
    self._results['SYS_PLATFORM'].append(system)
    if lst_gpus:
      self._results['GPU_NAME'] = lst_gpus[0]['NAME']
      self._results['GPU_MEMORY'] = lst_gpus[0]['TOTAL_MEM']
    else:
      self._results['GPU_NAME'].append('N/A')
      self._results['GPU_MEMORY'].append('N/A')
    return  
  
  def _load_images(self):
    def _load(path_images):
      lst_names = list(sorted(os.listdir(path_images)))
      lst_paths = [os.path.join(path_images, x) for x in lst_names]
      lst_imgs = [cv2.imread(x) for x in lst_paths]
      return lst_names, lst_imgs
    #enddef
    
    try:
      import cv2
      if self.DEBUG:
        self.log.p('Loading images classification')
      path_images = os.path.join(
        self.log.get_data_folder(), 
        self.config_data[ct.PATH_IMAGES_CLASSIFICATION]
        )
      lst_names, lst_imgs = _load(path_images)
      self.dct_imgs_classification = dict(zip(lst_names, lst_imgs))
      
      
      path_images = os.path.join(
        self.log.get_data_folder(), 
        self.config_data[ct.PATH_IMAGES_DETECTION]
        )
      lst_names, lst_imgs = _load(path_images)
      self.dct_imgs_detection = dict(zip(lst_names, lst_imgs))
      
      if self.DEBUG:
        self.log.p('{} classification images loaded'.format(len(self.dct_imgs_classification)))
        self.log.p('{} detection images loaded'.format(len(self.dct_imgs_detection)))
    except:
      self.log.p('Images could not be loaded', 'r')
    return
  
  def _run_pytorch(self):
    def _predict(th_graph):
      timer_name = th_graph._timer_name(ct.TIMER_SESSION_RUN)
      self.log.reset_timer(timer_name)
      
      device = next(th_graph.model.parameters()).device
      self.log.p('Pytorch model running on {}'.format(device.type.upper()))
      
      #infer
      lst_imgs = list(self.dct_imgs_classification.values())
      self.log.p('Running inference ...')
      for i in range(NR_RUN_INF):
        dct_inf = th_graph.predict(lst_imgs)
      
      #check results
      lst_inf = dct_inf[ct.INFERENCES]
      lst_classes = [x['TYPE'] for x in lst_inf]
      lst = list(zip(lst_classes, CLF_OBJECTS))
      s = ''
      for i,x in enumerate(lst):
        s+= '{}/{}'.format(x[0], x[1])
        if i < len(lst) - 1:
          s+= ', '
      # self.log.p('Classification results: {}'.format(s))
      
      nr_missed = len(set(CLF_OBJECTS) - set(lst_classes))
      nr_identified = len(CLF_OBJECTS) - nr_missed
      self.log.p(
        str_msg='{}/{} images correctly classified'.format(nr_identified, len(self.dct_imgs_classification)),
        color='g'
        )
            
      total_time = self.log.get_timing(timer_name)
      time_per_frame = total_time / len(self.dct_imgs_classification)
      self.log.p(
        '{:.4f}s per frame, {:.4f}s per batch ({})'.format(
          time_per_frame,
          total_time,
          len(self.dct_imgs_classification)
        ),
        color='g'
        )
      return total_time
    
    try:
      try:
        import torch as th
        self._results['TH_VER'].append(th.__version__)
      except Exception as e:
        self.log.p('Pytorch not loaded. Please check if Pytorch is configured \
                   in the current environment')
        self.log.p('Pytorch tests cannot run on the current environment.')
        self.log.p('Exception: {}'.format(str(e)), color='r')
        return
      #end try-except
      
      from inference import PytorchGraph
      self.log.p('Pytorch v{} working'.format(th.__version__))

      #loading graph    
      th_graph = PytorchGraph(
        log=self.log,
        config_graph=self._cfg_inf[ct.PYTORCH]
        )      
      device = next(th_graph.model.parameters()).device
      device_type = device.type.upper()
      
      device_name = 'GPU' if device_type == 'CUDA' else 'CPU'              
      
      self.log.p('Memory status before Pytorch run on {}'.format(device_name))
      self.app_monitor.log_gpu_info()
      total_time = _predict(th_graph)
      self.log.p('Memory status after Pytorch run on {}'.format(device_name))
      self.app_monitor.log_gpu_info()
      
      if device.type.upper() == 'CUDA':
        self._results['TH_GPU_TIME'].append(total_time)
      else:
        self._results['TH_CPU_TIME'].append(total_time)
        self._results['TH_GPU_TIME'].append('N/A')
      #endif
      
      if device.type.upper() == 'CUDA':
        th_graph.DEVICE = th.device('cpu')
        th_graph.model.to(th_graph.DEVICE)
        self.log.p('Memory status before Pytorch run on CPU')
        self.app_monitor.log_gpu_info()
        total_time = _predict(th_graph)
        self.log.p('Memory status after Pytorch run on CPU')
        self.app_monitor.log_gpu_info()
        self._results['TH_CPU_TIME'].append(total_time)
      #endif
      
      del th_graph.model
      del th_graph
      import gc
      gc.collect()
      th.cuda.empty_cache()
      
      self._pytorch_ok = True
    except:
      str_e = traceback.format_exc()
      self.log.p(
        'Exception encountered in Pytorch step: {}'.format(str_e), 
        color='r'
        )
    return
  
  def _run_tensorflow(self):
    def _predict(tf_graph):
      timer_name = tf_graph._timer_name(ct.TIMER_SESSION_RUN)
      self.log.reset_timer(timer_name)
      
      #infer
      lst_imgs = list(self.dct_imgs_detection.values())
      self.log.p('Running inference ...')
      for i in range(NR_RUN_INF):
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
      
      total_time = self.log.get_timing(timer_name)
      time_per_frame = total_time / len(self.dct_imgs_detection)
      self.log.p(
        str_msg='{:.4f}s per frame, {:.4f}s per batch ({})'.format(
          time_per_frame,
          total_time,
          len(self.dct_imgs_detection)
          ),
        color='g'
        )
      return total_time
    
    try:
      try:
        import tensorflow as tf
        # tf.debugging.experimental.enable_dump_debug_info('dump.txt')
        major = int(tf.__version__[0])
        minor = int(tf.__version__[2])
        assert major == 2 and int(minor) >= 0, 'Environment needs tensorflow >= 2.1.0, found: {}'.format(tf.__version__)
        self._results['TF_VER'].append(tf.__version__)
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
      lst_gpus = self.log.get_gpu()
      has_gpu = len(lst_gpus) > 0      
      
      if has_gpu:
        tf_graph = TensorflowGraph(
        log=self.log,
        config_graph=self._cfg_inf[ct.TENSORFLOW],
        on_gpu=True
        )
        self.log.p('Tensorflow model running on GPU')
        self.log.p('Memory status before Tensorflow run on GPU')
        self.app_monitor.log_gpu_info()
        total_time = _predict(tf_graph)
        self.log.p('Memory status after Tensorflow run on GPU')
        self.app_monitor.log_gpu_info()
        self._results['TF_GPU_TIME'].append(total_time)
      else:
        self._results['TF_GPU_TIME'].append('N/A')
      #endif
      
      tf_graph = TensorflowGraph(
        log=self.log,
        config_graph=self._cfg_inf[ct.TENSORFLOW],
        on_gpu=False
        )
      self.log.p('Tensorflow model running on CPU')
      self.log.p('Memory status before Tensorflow run on CPU')
      self.app_monitor.log_gpu_info()
      total_time = _predict(tf_graph)
      self.log.p('Memory status after Tensorflow run on CPU')
      self.app_monitor.log_gpu_info()
      self._results['TF_CPU_TIME'].append(total_time)
      
      self._tensorflow_ok = True
    except:
      str_e = traceback.format_exc()
      self.log.p(
        str_msg='Exception encountered in Tensorflow step: {}'.format(str_e),
        color='r'
        )
    return
  
  def _check_opencv(self):
    try:
      import cv2
      self._results['OPENCV_VER'].append(cv2.__version__)
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
      df = pd.DataFrame(self._results)
      fn = os.path.basename(self.log.log_file) + '.csv'
      full_path = os.path.join(self.log.get_output_folder(), fn)
      self.log.save_dataframe(
        df=df,
        fn=fn,
        folder='output'
        )
      self.log.p(
        str_msg='Results obtained for the current run can be found in: {}'.format(full_path),
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
    
    
    
    
    
    
    
    
    
    