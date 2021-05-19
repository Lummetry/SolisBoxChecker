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

from libraries.logger import Logger
from libraries.generic_obj import LummetryObject

class SimpleClient(LummetryObject):
  def __init__(self):
    return
  
  def send_alive_signal(self):
    # write "ALIVE" shared memory key
    return
  
  def _pop_data(self):
    # Redis.POP
    return
  
  def _push_data(self, data):
    # Redis.WRITE
    return
  
  def _process_data(self, data):
    self._push_data(data)
  
  def main_loop(self):
    self.send_alive_signal()
    while not self._done_processing:
      data = self._pop_data()
      if data is not None:
        self._process_data(data)
        
if __name__ == '__main__':
  pass
        