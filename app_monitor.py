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
import numpy as np
import constants as ct
from collections import deque
import traceback
import platform

from libraries_pub import LummetryObject

__VER__ = '0.3.1.0'

MIN_GLOBAL_INCREASE_GB = 0.010
MIN_PROCESS_INCREASE_MB = 100

class ApplicationMonitor(LummetryObject):
  def __init__(self, log, **kwargs):
    self.avail_memory_log = deque(maxlen=1000)
    self.process_memory_log = deque(maxlen=1000)
    self.gpu_memory_log = deque(maxlen=1000)
    self.start_avail_memory = None
    self._max_avail_increase = MIN_GLOBAL_INCREASE_GB
    self._max_process_increase = MIN_PROCESS_INCREASE_MB
    self.start_process_memory = None
    self.alert = False
    self.alert_avail = False
    self.alert_process = False
    self._done_first_smi_error = False
    self.message = ""
    self.version = __VER__
    super().__init__(log=log, **kwargs)
    return
  
  def log_status(self, color):
    if len(self.avail_memory_log) < 2 and len(self.process_memory_log) < 2:      
      return
    delta_avail_start = self.start_avail_memory - self.avail_memory_log[-1]
    delta_avail_last = self.avail_memory_log[-2] - self.avail_memory_log[-1]
    delta_process_start = self.process_memory_log[-1] - self.start_process_memory
    delta_process_last =  self.process_memory_log[-1] -  self.process_memory_log[-2]
    
    np_a = np.array(self.process_memory_log)
    avg = np.mean(np_a[1:] - np_a[:-1])

    self.P("=================================================================", color=color)
    self.P("Memory status{}".format(
      ' alert:' if color[0] == 'r' else ':'), color=color)
    self.P("  Global mem avail start: {:0,.3f} GB".format(self.start_avail_memory), color=color)
    self.P("  Global mem avail now:   {:0,.3f} GB".format(self.avail_memory_log[-1]), color=color)
    self.P("  Global mem decrease:    {:0,.3f} GB".format(delta_avail_start), color=color)
    self.P("  Global mem last dec:    {:0,.3f} GB".format(delta_avail_last), color=color)
    self.P(" -------------------------------------------------------------", color=color)
    self.P("  Process mem start:      {:0,.3f} GB".format(self.start_process_memory / 1024), color=color)
    self.P("  Process mem now:        {:0,.3f} GB".format(self.process_memory_log[-1] / 1024), color=color)
    self.P("  Process mem increase:   {:0,.3f} GB".format(delta_process_start / 1024), color=color)
    self.P("  Process mem last inc:   {:0,.3f} GB".format(delta_process_last / 1024), color=color)
    self.P("  Process mem mean inc:   {:0,.3f} GB".format(avg / 1024), color=color)
    self.P(" -------------------------------------------------------------", color=color)
    self.P("  Global memory loss due to out-of-process factors:  {:0,.3f} GB".format(
      delta_avail_start - (delta_process_start / 1024)), color=color)
    self.P("  Number of readings: {}".format(len(self.avail_memory_log)), color=color)
    self.P("=================================================================", color=color)
    return
  
  def send_message(self, str_message):
    self.add_message(str_message)
    return

  
  def add_message(self, str_message):
    self.message = self.message + '\n' + str_message
    return
      
  def log_avail_memory(self):
    avail_memory = self.log.get_avail_memory()
    self.avail_memory_log.append(avail_memory)
    if len(self.avail_memory_log) == 1:
      self.start_avail_memory = avail_memory
    else:
      if avail_memory < self.start_avail_memory:
        # check for abnormal increase
        delta_start = self.start_avail_memory - avail_memory        
        if delta_start >= (self._max_avail_increase + MIN_GLOBAL_INCREASE_GB):
          self.alert_avail = True
          self._max_avail_increase = delta_start
        else:
          self.alert_avail = False

    return avail_memory


  def log_process_memory(self):
    process_memory = self.log.get_current_process_memory(mb=True)
    self.process_memory_log.append(process_memory)
    if len(self.process_memory_log) == 1:
      self.start_process_memory = process_memory
    else:
      if process_memory > self.start_process_memory:
        # check for abnormal increase
        delta_start = process_memory - self.start_process_memory 
        if delta_start > (self._max_process_increase + MIN_PROCESS_INCREASE_MB):
          self.alert_process = True
          self._max_process_increase = delta_start
        else:
          self.alert_process = False

    return process_memory
  
  
  def gpu_info(self):
    """
    THIS FUNCTION WILL BE MOVED TO LOGGER AFTER SERIOS TUNING

    Returns
    -------
    lst_inf : list[dict]
      each element contains GPU info 
    """
    lst_inf = []
    try:
      import torch as th
      from pynvml.smi import nvidia_smi    

      name = th.cuda.get_device_name(0)

      nvsmi = nvidia_smi.getInstance()
      res = nvsmi.DeviceQuery('memory.free, memory.total, memory.used, utilization.gpu, temperature.gpu')

      dct_gpu = res['gpu'][0]
      mem_total = dct_gpu['fb_memory_usage']['total'] / 1024
      mem_allocated = dct_gpu['fb_memory_usage']['used'] / 1024
      gpu_used = dct_gpu['utilization']['gpu_util']
      gpu_temp = dct_gpu['temperature']['gpu_temp']
      gpu_temp_max = dct_gpu['temperature']['gpu_temp_max_threshold']
      lst_inf.append({
          'NAME': name,
          'TOTAL_MEM': mem_total,
          'ALLOCATED_MEM': mem_allocated,
          'GPU_USED' : gpu_used,
          'GPU_TEMP' : gpu_temp,
          'GPU_TEMP_MAX' : gpu_temp_max,
          })
    except:
      if not self._done_first_smi_error:
        self.P("ERROR: Please make sure you have both pytorch and pynvml in order to monitor the GPU",
               color='r')
        str_err = traceback.format_exc()
        self.add_message("Pytorch or NVidia-SMI issue:   \n\n{}".format(str_err))
        self._done_first_smi_error = True
        
    return lst_inf
  
  def log_gpu_info(self):
    lst_inf = self.gpu_info()
    for dct in lst_inf:
      self.log.p('GPU status:')
      for k,v in dct.items():
        self.log.p(' {}: {}'.format(k, v))
    return
    
    
    
    
  