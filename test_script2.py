# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:33:36 2014

@author: JDD46
"""

from summarytools import *
from datetime import datetime
test = open_dataset('Coke.pkl')
startTime = datetime.now()
scores = test.select_summary()

print 'Elapsed time:', (datetime.now()-startTime)