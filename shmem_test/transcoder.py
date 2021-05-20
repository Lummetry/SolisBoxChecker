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
import mmap
import os
import pickle

from libraries_pub.logger import Logger
from libraries_pub.generic_obj import LummetryObject



from shmem_test.streamer import SimpleStreamingPipeline

from shmem_test.config import AVAIL_STREAMS, TRANSCODER_SUFIX, AI_SUFIX, BUFFER_KEY, HEIGHT_KEY, WIDTH_KEY, CHANNEL_KEY



class SimpleStreamCalbackHelper:
  def __init__(self, stream_name, message_queue, log):
    self.name = stream_name
    self.log = log
    self._message_queue = message_queue
    self._can_send = True
    self._data_counter = 0
    self._received = []
    self._sent = []    
    self._init_shmem()
    return
  
  def log_info(self, str_msg):
    self._message_queue.append("    [SVR-CB-{}] {}".format(self.name,str_msg))
    return
  
  def _pseudo_shmem_init(self):
    self._fn_write = os.path.join(self.log.get_data_folder(), self.name + TRANSCODER_SUFIX)
    self._fn_read = os.path.join(self.log.get_data_folder(), self.name + AI_SUFIX)
  
  def _pseudo_shmem_write(self, data):
    with open(self._fn_write, 'wb') as fh:
      pickle.dump(data, fh)
    return
  
  def _pseudo_shmem_read(self):
    data = None
    if os.path.isfile(self._fn_read):
      try:
        with open(self._fn_read, 'rb') as fh:
          data = pickle.load(fh)
        os.remove(self._fn_read)
      except:
        pass
    return data
  
  
  def _init_shmem(self):
    # Redis initialize THIS stream shmem
    self._pseudo_shmem_init()
    return
    
  
  def _write_to_shmem(self, data):
    # Redis write to "send" location
    self._pseudo_shmem_write(data)
    pass
  
  def _pop_from_shmem(self):
    # Readis read from "incoming location
    data = self._pseudo_shmem_read()
    return data
  
  def start_timer(self, str_name):
    self.log.start_timer(self.name + '_' + str_name)
    return
  
  def stop_timer(self, str_name):
    self.log.stop_timer(self.name + '_' + str_name)
    return

  def _send_data_to_client(self, data):
    self._data_counter += 1
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
    if data is not None and type(data) == np.ndarray:
      self._received.append(data.ravel()[0])
    return data
  
  def __unpack_streamer_payload(self, data):
    # do required processing on data from streaming pipeline
    sleep(0.005)
    dct_data = {
      BUFFER_KEY : data['B'],
      HEIGHT_KEY : data['HH'],
      WIDTH_KEY : data['WW'],
      CHANNEL_KEY : data['CC']
      }
    return dct_data

    
  
  def _callback_for_streamer(self, data):
    self.start_timer('read_from_stream')    
    data = self.__unpack_streamer_payload(data)    
    self.stop_timer('read_from_stream')

    # we push data to client if we can
    if self.can_send_to_client():
      self._send_data_to_client(data)
      sent = len(self._sent)
      if (sent % 100) == 0:
        self.log_info('Delivered {} frames from current stream'.format(sent))
    
    
    # "from_client" is NOT sync-ed (same package) with above `np_img`
    from_client = self._read_data_from_client()
    if from_client is not None:
      self._process_data_from_client(from_client)
      recv = len(self._received)
      if (recv % 100) == 0:
        self.log_info('Received {} frames based on current stream. Missing {} distinct frames'.format(
          recv,
          len(set(self._sent) - set(self._received)),
          ))
      
    return
  
    
  def can_send_to_client(self):
    # here we can check a shared memory key that informs if client 
    # is announced
    return self._can_send
  
  def _process_data_from_client(self, data):
    # this function processes the data from client
    # eg. concatentates frames and provides outgoing ML processed stream    
    return  
  
      

class SimpleServer(LummetryObject):
  def __init__(self, **kwargs):
    self.messages = deque(maxlen=100)
    self.streams = []
    super().__init__(**kwargs)
    return
    
    
  def start_new_stream(self, name):
    self.P("[SVR] Starting new stream '{}'".format(name))
    new_callback_handler = SimpleStreamCalbackHelper(
      stream_name=name, 
      message_queue=self.messages,
      log=self.log
      )
    new_stream = SimpleStreamingPipeline(      
      stream_name=name, 
      callback=new_callback_handler._callback_for_streamer,
      message_queue=self.messages,
      log=self.log
      )
    new_thread = Thread(target=new_stream._stream_loop)
    new_thread.start()
    self.streams.append({
      'STREAM_OBJECT'   : new_stream,
      'CALLBACK_OBJECT' : new_callback_handler,
      }      
      )
    self.P("[SVR] Stream started.")
    return
  
  def dump_log(self):
    if len(self.messages) > 0:
      msg = self.messages.popleft()
      self.P(msg)
    
  def stop_streams(self):
    for stream in self.streams:
      stream['STREAM_OBJECT'].stop_stream()
      cbobj = stream['CALLBACK_OBJECT']
      self.P("Stream {} delivered {} and received {} frames. Missing {} specific frames.".format(
        cbobj.name,
        len(cbobj._sent),
        len(cbobj._received),
        len(set(cbobj._sent) - set(cbobj._received)),
        ))
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
  
  # we setup custom logging system
  l = Logger('SHMSVR', base_folder='.', app_folder='_cache', TF_KERAS=False)  
  
  run_time = MAX_RUN_TIME * 60
  elapsed = 0  
  start_time = time()
  
  
  # start server and a few streams
  eng = SimpleServer(log=l)

  l.P('Running for {} seconds'.format(run_time))

  for stream_name in AVAIL_STREAMS:
    eng.start_new_stream(stream_name)
  
  show_intervals = run_time // 10
  while elapsed < run_time:
    # we just wait 1 sec in main thread then we dump the log
    sleep(1)
    eng.dump_log()
    elapsed = int(time() - start_time)
    if (elapsed % show_intervals) == 0:
      l.P("Running for another {}s".format(run_time - elapsed))
  
  eng.shutdown()

  l.show_timers(show_levels=False) # we cant show imbrications due to parallelism of operations
 
  