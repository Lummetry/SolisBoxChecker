# -*- coding: utf-8 -*-
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
import pickle
import os
from time import sleep

from libraries_pub.logger import Logger
from libraries_pub.generic_obj import LummetryObject

from shmem_test.config import AVAIL_STREAMS, TRANSCODER_SUFIX, AI_SUFIX, BUFFER_KEY, HEIGHT_KEY, WIDTH_KEY, CHANNEL_KEY


class SimpleClient(LummetryObject):
  def __init__(self, **kwargs):
    self.streams = {}
    self._done_processing = False
    super().__init__(**kwargs)
    return
  
      
  def _pseudo_shmem_init(self):    
    for stream_name in AVAIL_STREAMS:
      pseudo_shmem_incoming = os.path.join(
        self.log.get_data_folder(),
        stream_name + TRANSCODER_SUFIX
        )
      pseudo_shmem_outgoing = os.path.join(
        self.log.get_data_folder(),
        stream_name + AI_SUFIX
        )
      
      self.streams[stream_name] = {
        'READ'  : pseudo_shmem_incoming,
        'WRITE' : pseudo_shmem_outgoing,
        'RECEIVED' : [],
        'SENT'     : [],
        }
    return
        
  def _pseudo_shmem_write(self, data, stream_name):
    fn_write = self.streams[stream_name]['WRITE']
    try:
      with open(fn_write, 'wb') as fh:
        pickle.dump(data, fh)
        success = True
    except:
      success = False
    if success:
      self.streams[stream_name]['SENT'].append(data.ravel()[0])
    return success
  
  def _pseudo_shmem_read(self, stream_name):
    fn_read = self.streams[stream_name]['READ']
    data = None
    if os.path.isfile(fn_read):
      try:
        with open(fn_read, 'rb') as fh:
          data = pickle.load(fh)
        os.remove(fn_read)
      except:
        pass
    return data
  
    
  
  def send_alive_signal(self):
    # write "ALIVE" shared memory key
    return
  
  def _init_shmem(self):
    # Redis initialize sh memory locations
    self._pseudo_shmem_init()
    self.P("SHMEM initialized")
    return
  
  def _pop_data(self, stream_name):
    # Redis.POP
    self.log.start_timer('pop_data')
    data = self._pseudo_shmem_read(stream_name)
    self.log.stop_timer('pop_data')    
    return data
  
  def _push_data(self, data, stream_name):
    # Redis.WRITE
    self.log.start_timer('push_data')
    res = self._pseudo_shmem_write(data, stream_name)
    self.log.stop_timer('push_data')    
    return
  
  def _process_data(self, data, stream_name):
    img_h = data[HEIGHT_KEY]
    img_w = data[WIDTH_KEY]
    img_c = data[CHANNEL_KEY]
    img_data = data[BUFFER_KEY]
    np_data = np.frombuffer(img_data, dtype=np.uint16) # uint16 due to fake data
    vvv = np_data[0]
    np_img = np_data.reshape((img_h, img_w, img_c))
    self.streams[stream_name]['RECEIVED'].append(vvv)
    rcv = self.streams[stream_name]['RECEIVED']
    cnt = len(rcv)
    if cnt == 1:
      self.P("Data receiving from {} started".format(stream_name))
    if (cnt % 100) == 0:
      lost = sorted(list(set(np.arange(1, vvv+1)) - set(rcv)))
      self.P("So far received {} frames from stream: {}. Looks like lost {} frames in process {}.".format(
        cnt,
        stream_name,
        len(lost), 
        "(eg. {}...)".format(lost[:10]) if len(lost) > 0 else "",
        ))
    return np_img
  
  def main_loop(self):
    self.P("Starting AI main loop. Press Ctrl-C to stop.")
    self._init_shmem()
    self.send_alive_signal()
    while not self._done_processing:
      sleep(0.0005) # for process yielding
      for stream_name in self.streams:
        data = self._pop_data(stream_name)
        if data is not None:
            np_img = self._process_data(data, stream_name)
            # now we send back directly the np_img just for testing purposes
            # this can be replaced with binary value and reconstructed in C/C++
            self._push_data(np_img, stream_name)
    self.P("Closing AI main loop.")
    return
        
if __name__ == '__main__':
  l = Logger('VB', base_folder='.', app_folder='_cache')
  vb = SimpleClient(log=l)
  vb.main_loop() 
  
        