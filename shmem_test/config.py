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
  
IMPORTANT: 
  This config file must be replaced by a "STATUS" shared mem between 
transcoder and AI module where the current available streams will be described

  
"""

STREAM_1 = 'S1'
STREAM_2 = 'S2'
STREAM_3 = 'S3'

AVAIL_STREAMS = [
  STREAM_1,
  STREAM_2,
  STREAM_3,
  ]


TRANSCODER_SUFIX = '_WRITE'
AI_SUFIX = '_READ'

BUFFER_KEY = 'BUFFER'
HEIGHT_KEY = 'H'
WIDTH_KEY = 'W'
CHANNEL_KEY = 'C'