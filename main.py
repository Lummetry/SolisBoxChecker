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

from app import VaporBoxCheck
from libraries_pub import Logger

if __name__ == '__main__':
  log = Logger(
    lib_name='VBC', 
    config_file='config.txt',
    TF_KERAS=False
    )
  
  vbc = VaporBoxCheck(log=log, DEBUG=True)
  vbc.run()
  