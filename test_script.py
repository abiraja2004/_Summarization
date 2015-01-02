# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 18:16:09 2014

@author: JDD46
"""

from summarytools import *
from datetime import datetime
#start = datetime.now()
#test = DataSet(setname='frozen', pos_tag=0)
#test.extract_from('frozenfood_small.csv', 7)
#print 'Setup:', (datetime.now()-start)
#startTime = datetime.now()
#test.extract_data()
#print 'Extraction:', (datetime.now()-startTime)
#startTime = datetime.now()
#test.calc_scores()
#print 'Scoring:', (datetime.now()-startTime)
test = open_dataset('frozen.pkl')
#startTime = datetime.now()
#test.build_svec_matrix()
#print 'Matrix Construction:', (datetime.now()-startTime)
startTime = datetime.now()
test.remove_rt()
print 'RT Removal:', (datetime.now()-startTime)    
#print 'Total time:', (datetime.now()-start)
