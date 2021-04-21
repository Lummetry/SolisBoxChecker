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
from time import sleep
from multiprocessing.connection import Listener
from threading import Thread
from collections import deque

from config import SERVER_PORT, SERVER_URI, SERVER_PASS, MAX_QUEUE


class TranscoderSimulator:
  """
  simple server that handles just one client connection
  """
  def __init__(self, server_address=None, server_port=None, password=None):    
    self._server_port = SERVER_PORT if server_port is None else server_port
    self._server_address = SERVER_URI if server_address is None else server_address
    self._password = SERVER_PASS if password is None else password
    self._to_send = deque(maxlen=MAX_QUEUE)
    self._processed = deque(maxlen=MAX_QUEUE)
    self._log = deque(maxlen=MAX_QUEUE)
    self._setup()
    return
  
  
  def add_to_send_queue(self, buffer):
    self._to_send.append(buffer)
    return
  
  def get_processed_data(self):
    """
      This function pops and returns the FIFO data 
    """
    ret = None
    if len(self._processed) > 0:
      ret= self._processed.popleft()
    return ret
  
  def has_to_send(self):
    return len(self._to_send) > 0
  
  
  def get_log(self):
    """
    this function is just a helper that generates last log entry of the server as
    the server does not print/output logs by itself for thread safety. Similar I/O 
    should be handled in the same way.
    """
    ret = None
    if len(self._log) > 0:
      ret = self._log.popleft()
    return ret
  
  
  def _add_log(self, s):
    self._log.append(s)
    return
    
  
  def _setup(self):
    """
    setup server and thread - no connection here in order not to block python process
    """
    self._listener = Listener(
      address=(self._server_address, self._server_port),
      authkey=self._password
      )
    self._thread = Thread(target=self._runner)
    return
      
  
  def _runner(self):
    """
      This code runs in parallel with the main process and it handles both sending
      and added new payloads to internal buffers that can be consumed from external
      threads
    """
    self._add_log("SVR: listening for connections...")
    self._conn = self._listener.accept()
    self._add_log("SVR: connection accepted")
    self._done = False
    n_snd, n_rcv = 0, 0
    MSG_TO_DISPLAY = 5
    while not self._done:
      # check if something is available
      if self._conn.poll():
        buff = self._conn.recv()
        n_rcv += 1
        self._processed.append(buff)
        if (n_rcv % MSG_TO_DISPLAY) == 0:
          self._add_log('SVR: received processed buffer ({})'.format(n_rcv))
          
      # now we send a package if any is available from a external process
      # that is main process in our case
      if self.has_to_send():
        n_snd += 1
        buff_to_send = self._to_send.popleft()
        self._conn.send(buff_to_send)    
        if (n_snd % MSG_TO_DISPLAY) == 0:
          self._add_log('SVR: delivered raw buffer ({})'.format(n_snd))
      sleep(0.05)
    return        
    
  
  def run(self):
    self._thread.start()
    
    
  def stop(self):
    self._done = True
    self._listener.close()
    return
  
  
if __name__ == '__main__':
  
  INPUT_CHECK = 100
  DISPLAY_EVERY = 20
  
  def __log(s):
    print(s, flush=True)
   
  # We simulate the generation of frames in the main thread
  
  __log("Starting server...")
  svr = TranscoderSimulator()
  svr.run()
  n_frames = 0
  n_proc_frames = 0
  while True: 
    # we run until Ctrl-C
        
    # we check if there is log available
    s = svr.get_log()
    if s is not None:
      __log(s)
    np_img = np.ones(shape=(720, 1280, 3)).astype(np.uint8) * INPUT_CHECK
    n_frames += 1
    id_frame = n_frames + 9191 # generated some fake id :)
    dct_buff = {
      'DATA' : np_img,
      'ID' : id_frame
      }
    svr.add_to_send_queue(dct_buff)    
    if (n_frames % 20) == 0:
      __log("Generated {:>4} frames so far. Last generated frame ID: {:>4}".format(
        n_frames, id_frame))
    sleep(0.1) # wait 1000 msec
    
    # now lets see if we have some data that has been processed
    dct_rcv_buff = svr.get_processed_data()
    if dct_rcv_buff is not None:
      np_rcv_img = dct_rcv_buff['DATA']
      id_rcv =  dct_rcv_buff['ID']
      vv = np_rcv_img[0,0,0]
      if vv != INPUT_CHECK:
        n_proc_frames += 1
        if (n_proc_frames % DISPLAY_EVERY) == 0:          
          __log("Received {:>4} processed frames so far. Last frame ID:{:>4}".format(
            n_proc_frames, id_rcv))
      
      
    
    
    
  
    
    