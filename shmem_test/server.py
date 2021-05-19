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
from threading import Thread
from time import sleep

class SimpleServer:
  def __init__(self,):
    self.name = 'SHSVR'
    self._can_send = False
    self._stop_data = False
    self._data_counter = 0
    return
  
  def can_send_to_client(self):
    # here we can check a shared memory key that informs if client 
    # is announced
    return self._can_send
  
  def _send_data(self):
    # now we write the shared memory location
    return
  
  def _process_data_from_client(self, data):
    # this function processes the data from client
    # eg. concatentates frames and provides outgoing ML processed stream
    return
  
  def _stream_loop(self):
    # one thread for each stream
    while not self._stop_data:
      # we aquire data from stream
      sleep(0.01) # we wait 10 milisec to simulate 100 fps
      self._data_counter += 1
      np_img = np.ones((720, 1280, 3)) * self._data_counter
            
      # we push data to client if we can
      if self.can_send_to_client():
        self._send_data(np_img)
      
      from_client = self._pop_client_data()
      # "from_client" is NOT sync-ed (same package) with above `np_img`
      if from_client is not None:
        self._process_data_from_client(from_client)
    

if __name__ == '__main__':
  pass