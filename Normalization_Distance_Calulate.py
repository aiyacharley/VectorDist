from __future__ import division
import os,sys
import numpy as np
import scipy.spatial.distance as scipydist
import string
import math

class Normalize: ### normalization methods
	vector = []
	def __init__(self,mylist):
		self.vector = mylist
	def MaxMin(self):
		maxval = max(self.vector)
		minval = min(self.vector)
		return [(i-minval)/(maxval - minval) for i in self.vector]
	def Z_Score(self):
		aver = np.average(self.vector)
		sigma = np.std(self.vector)
		return [(i-aver)/sigma for i in self.vector]
	def CountFrac(self):
		return [float(i)/sum(self.vector) for i in self.vector]
	def sigmoid(self):
		return [1.0 / (1 + np.exp(- float(x))) for x in self.vector]

class Distance: ### Distance between vectors
	p, q = [], []
	def __init__(self,list1,list2):
		self.p = list1
		self.q = list2
	def Manhattan(self):
		return np.sum(abs( np.mat(self.p) - np.mat(self.q) ))
	def Euclidean(self):
		return np.linalg.norm( np.mat(self.p) - np.mat(self.q) )
	def Chebyshev(self):
		return abs(np.mat(self.p) - np.mat(self.q)).max()
	def Cosine(self):
		return 1 - np.dot(self.p,self.q)/(np.linalg.norm(self.p)*np.linalg.norm(self.q))
	# KLD and JSD distance: p,q required probability distribution
	def KLD(self):
		p,q = zip(*filter(lambda (x,y): x!=0 or y!=0, zip(self.p,self.q))) # remove both p and q value equal 0
		p = p + np.spacing(1)						  
		q = q + np.spacing(1)
		dist = sum([_p * math.log(_p/_q,2) for (_p,_q) in zip(p,q)])
		return dist
	def JSD(self):
		p,q = zip(*filter(lambda (x,y): x!=0 or y!=0, zip(self.p,self.q)))
		M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
		p = p + np.spacing(1)						  
		q = q + np.spacing(1)
		M = M + np.spacing(1)
		return 0.5*Distance(p,M).KLD()+0.5*Distance(q,M).KLD()
	# Hamming and Jaccard distance: p,q required 0 or 1 vector
	def Hamming(self):
		smstr = np.nonzero(np.mat(self.p) - np.mat(self.q))
		return np.shape(smstr[0])[1]
	def Jaccard(self):
		matv = np.mat([self.p,self.q])
		return scipydist.pdist(matv,"jaccard")
	# Morisita-Horn overlap index
	
"""	   
p=np.ones(5)/5.0
#p = [2,3,5,4,2]
q=[0,0,0.5,0.2,0.3]

print Normalize(q).MaxMin()
print Normalize(q).Z_Score()
print Normalize(q).CountFrac()
print Normalize(q).sigmoid()

print Distance(p,q).Manhattan()
print Distance(p,q).Euclidean()
print Distance(p,q).Chebyshev()
print Distance(p,q).Cosine()
print Distance(p,q).KLD()
print Distance(p,q).JSD()
print help(Distance)
"""
