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


from time import sleep
from multiprocessing.connection import Client
from threading import Thread
from collections import deque

from config import SERVER_PORT, SERVER_URI, SERVER_PASS, MAX_QUEUE


class VaporBoxSimulator:
  """
  simple server that handles just one client connection
  """
  def __init__(self, server_address=None, server_port=None, password=None):    
    self._server_port = SERVER_PORT if server_port is None else server_port
    self._server_address = SERVER_URI if server_address is None else server_address
    self._password = SERVER_PASS if password is None else password
    self._log = deque(maxlen=MAX_QUEUE)
    
    # now we define a size 1 queue due to the fact that we would like to preserve only 
    # the newest frame and pop-it when it is processed
    self._buffer = deque(maxlen=1)
    self._setup()
    return
  
  
  
  def get_log(self):
    """
    this function is just a helper that generates last log entry of the server as
    the server does not print/output logs by itself
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
    setup client and thread - no connection here in order not to block python process
    """
    self._add_log('Connecting to {}:{}'.format(self._server_address, self._server_port))
    self._conn = Client(
      (self._server_address, self._server_port), 
      authkey=self._password
      )
    self._add_log("CLNT: connection established")
    
    # it is important to have a receiver thread that receives incoming data at 
    # a different rate that then thread that processes it and sends
    # this way we do not have overflows
    self._thread_rcv = Thread(target=self._runner_receiver)
    self._thread_snd = Thread(target=self._runner_sender)
    return
      
  def _runner_receiver(self):
    """
      This code runs in parallel with the main process
    """
    self._done = False
    n_rcv = 0
    MSG_TO_DISPLAY = 5
    zero_imgs = 0
    self._add_log('CLNT: startig receiver thread...')
    while not self._done:
      # check if something is available
      sleep(0.1)
      if self._conn.poll():        
        dct_buff = self._conn.recv()
        n_rcv += 1
        id_img = dct_buff['ID']
        zero_imgs = 0
        self._buffer.append(dct_buff)

        if (n_rcv % MSG_TO_DISPLAY) == 0:
          self._add_log('CLNT: received buffer ({}). Last ID: {}'.format(
            n_rcv, id_img))
      else:
        zero_imgs += 1
        if (zero_imgs % 200) == 0:
          self._add_log("CLNT: no messages for {} iters".format(zero_imgs))
                  
    return        

  def _runner_sender(self):
    """
      This code runs in parallel with the main process
    """
    self._done = False
    n_snd = 0
    MSG_TO_DISPLAY = 5
    zero_imgs = 0
    self._add_log('CLNT: startig processer thread...')
    while not self._done:
      # check if something is available
      sleep(0.1)
      if len(self._buffer) > 0:                
        dct_buff = self._buffer.pop()
        id_img = dct_buff['ID']
        dct_buff['DATA'] = dct_buff['DATA'] + 1000
        self._conn.send(dct_buff)
        n_snd += 1
        if (n_snd % MSG_TO_DISPLAY) == 0:
          self._add_log('CLNT: processd-send buffer ({}). Last ID: {}'.format(
            n_snd, id_img))
      else:
        zero_imgs += 1
        if (zero_imgs % 200) == 0:
          self._add_log("CLNT: no messages for {} iters".format(zero_imgs))
                  
    return        

    
  def run(self):
    self._thread_rcv.start()
    self._thread_snd.start()
    return
    
  def stop(self):
    self._done = True
    self._conn.close()
    return
  
  
if __name__ == '__main__':
  
  def __log(s):
    print(s, flush=True)
   
  # We simulate the generation of frames in the main thread
  
  __log("Starting client...")
  clnt = VaporBoxSimulator()
  clnt.run()
  DONE = False
  while not DONE:
    msg = clnt.get_log()
    if msg is not None:
      print(msg, flush=True)
