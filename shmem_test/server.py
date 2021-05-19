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
from time import time
from threading import Thread
from time import sleep
from collections import deque

from libraries_pub.logger import Logger
from libraries_pub.generic_obj import LummetryObject

class StreamRunner:
  def __init__(self, stream_name, message_queue, log):
    self.name = stream_name
    self.log = log
    self._message_queue = message_queue
    self._can_send = True
    self._stop_data = False
    self._data_counter = 0
    self._received = []
    self._sent = []
    return
  
  def log_info(self, str_msg):
    self._message_queue.append("[SVR-{}] {}".format(self.name,str_msg))
    return
  
  def _write_to_shmem(self, data):
    pass
  
  def _pop_from_shmem(self):
    return
  
  def start_timer(self, str_name):
    self.log.start_timer(self.name + '_' + str_name)
    return
  
  def stop_timer(self, str_name):
    self.log.stop_timer(self.name + '_' + str_name)
    return

  def _send_data_to_client(self, data):
    self._sent.append(self._data_counter)
    # now we write the shared memory location
    self.start_timer('write_to_shmem')
    self._write_to_shmem(data)
    self.stop_timer('write_to_shmem')
    return
  
  def _read_data_from_client(self):
    self.start_timer('read_to_shmem')
    data = self._pop_from_shmem()
    self.stop_timer('read_to_shmem')
    self._received.append(data.ravel()[0])
    return data
  
  def _low_level_stream_read(self):
    self._data_counter += 1
    data = np.ones((720, 1280, 3)) * self._data_counter     
    sleep(0.01) # we wait 10 milisec to simulate 100 fps
    return data
  
  def _read_from_stream(self):
    self.start_timer('read_from_stream')    
    data = self._low_level_stream_read()    
    self.stop_timer('read_from_stream')
    return data
    
  def can_send_to_client(self):
    # here we can check a shared memory key that informs if client 
    # is announced
    return self._can_send
  
  def _process_data_from_client(self, data):
    # this function processes the data from client
    # eg. concatentates frames and provides outgoing ML processed stream    
    return  
  
  def stop_stream(self):
    self._stop_data = True
    return
  
  def _stream_loop(self):
    # one thread for each stream
    self.log_info('Starting data aquisition and delivery')
    while not self._stop_data:
      # we aquire data from stream
      np_img = self._read_from_stream()
            
      # we push data to client if we can
      if self.can_send_to_client():
        self._send_data(np_img)
        if (self._data_counter % 100) == 0:
          self.log_info('Delivered {} frames from current stream'.format(self._data_counter))
      
      from_client = self._pop_from_shmem()
      # "from_client" is NOT sync-ed (same package) with above `np_img`
      if from_client is not None:
        self._process_data_from_client(from_client)
      
    # now we received the stop signal
    self.log_info('Closing stream {}'.format(self.name))
    self.log_info('Delivered {} frames, received {} frames. Missing {} frames'.format(
      len(self._sent),
      len(self._received),
      len(set(self._sent) - set(self._received)),
      ))
    
    
      

class SimpleServer(LummetryObject):
  def __init__(self, **kwargs):
    self.messages = deque(maxlen=100)
    self.streams = []
    super().__init__(**kwargs)
    return
    
    
  def start_new_stream(self, name):
    self.P("[SVR] Starting new stream '{}'".format(name))
    new_stream = StreamRunner(stream_name=name, message_queue=self.messages)
    new_thread = Thread(target=new_stream._stream_loop)
    new_thread.start()
    self.streams.append(new_stream)
    self.P("[SVR] Stream started.")
    return
  
  def dump_log(self):
    if len(self.messages) > 0:
      msg = self.messages.popleft()
      self.P(msg)
    
  def stop_streams(self):
    for stream in self.streams:
      stream.stop_stream()
    return
  
  def shutdown(self, wait_time=10):
    self.stop_streams()
    
    # we wait 10 sec for cleanup
    shutdown_time = 0
    start_shutdown = time()
    while shutdown_time <= wait_time:
      sleep(1)
      self.dump_log()
      shutdown_time = time() - start_shutdown
    self.P("Server stopped.")  
    


if __name__ == '__main__':
  
  MAX_RUN_TIME = 2 # we run continuously for MAX_RUN_TIME minutes  
  N_STREAMS = 3 # we have 3 streams in same time
  
  # we setup custom logging system
  l = Logger('SHMSVR', base_folder='.', app_folder='_cache', TF_KERAS=False)  
  
  run_time = MAX_RUN_TIME * 60
  elapsed = 0  
  start_time = time()
  
  # start server and a few streams
  eng = SimpleServer(log=l)
  for i in range(N_STREAMS):
    eng.start_new_stream('S{}'.format(i+1))
  
  show_intervals = run_time // 10
  while elapsed <= run_time:
    # we just wait 1 sec in main thread then we dump the log
    sleep(1)
    eng.dump_log()
    elapsed = int(time() - start_time)
    if (elapsed % show_intervals) == 0:
      l.P("Running for another {}s".format(run_time - elapsed))
  
  eng.shutdown()
  l.show_timings()
 
  