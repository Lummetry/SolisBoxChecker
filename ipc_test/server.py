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
from multiprocessing.connection import Listener
from threading import Thread

from .config import SERVER_PORT, SERVER_URI, SERVER_PASS


class SimpleServer:
  def __init__(self, server_address=None, server_port=None, password=None):    
    self._server_port = SERVER_PORT if server_port is None else server_port
    self._server_address = SERVER_URI if server_address is None else server_address
    self._password = SERVER_PASS if password is None else password
    self._setup()
    return
    
    
  def _setup(self):
    self._listener = Listener(self._address, authkey=self._password)
    self._thread = Thread(target=self._runner)
    return
    
    
  def _runner(self):
    
    
    
  def run(self):
    self._thread.run()
    