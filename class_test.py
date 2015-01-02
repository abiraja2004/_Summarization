# -*- coding: utf-8 -*-
"""
Created on Fri Dec 05 10:21:28 2014

@author: JDD46
"""

class A(object):
    def __init__(self):
        self.count = 20
        self.L = [1, 2, 3, 4]
        self.Blist = []
        
    def newB(self):
        b = B()
        self.Blist.append(b)
    
    def call(self, index):
        b[index].addAcount()
        b[index].add2Alist()
        
        
class B(object):
    count = 0
    def __init__(self):
        self.id = B.count
        B.count += 1
        
    def addAcount(self):
        
        
    def add2Alist(self):
        
    