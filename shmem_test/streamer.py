# -*- coding: utf-8 -*-
"""
Copyright 2019-2021 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


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
@created on: Thu May 20 10:33:01 2021
@created by: damia
"""

import numpy as np
from time import sleep

class SimpleStreamingPipeline:
  """
  This object simulates the GStreamer
  """
  def __init__(self, stream_name, callback, message_queue, log):
    self.name = stream_name
    self.log = log
    self._message_queue = message_queue
    self._stop_data = False
    self._data_counter = 0
    self._callback = callback
    return
  
  
  def log_info(self, str_msg):
    self._message_queue.append("  [SVR-PIPELINE-{}] {}".format(self.name,str_msg))
    return
  
  
  def stop_stream(self):
    self._stop_data = True
    return
  
  
  def _stream_loop(self):
    # one thread for each stream
    self.log_info('Starting data aquisition and generation')
    while not self._stop_data:
      # we get one frame from stream
      self._data_counter += 1
      # we use uint16 due to fake data
      np_img = (np.ones((720, 1280, 3)) * self._data_counter).astype(np.uint16)
      data = {
        'B' : np_img.tobytes(),
        'HH' : 720,
        'WW' : 1280,
        'CC' : 3,
        }
      
      sleep(0.01) # we wait 10 milisec to simulate 100 fps

      if (self._data_counter % 100) == 0:
        self.log_info('Generated {} frames from current stream'.format(self._data_counter))
      
      # we call the callback
      self._callback(data)
            
      
      
    # now we received the stop signal
    self.log_info('Closing stream {}'.format(self.name))
    