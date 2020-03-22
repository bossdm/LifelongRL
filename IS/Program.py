#
#
# from abc import abstractmethod
# from overrides import overrides
# import numpy as np
# from random import random
#
#
#
#
#
#
#
# class ProgramCell(object):
# 	@abstractmethod
# 	def generateInstruction(self):
# 		pass
#
# class DiscreteProgramCell(ProgramCell):
# 	def __init__(self, maxInstr, actions):
# 		self.timeUntilUnfrozen=0
# 		self.maxInstr = maxInstr
# 		assert len(actions) == maxInstr
# 		self.actions = actions
#
# 	@abstractmethod
# 	def frozenInstruction(self):
# 		pass
# 	@abstractmethod
# 	def normalInstruction(self):
# 		pass
# 	def generateInstruction(self):
# 		if self.timeUntilUnfrozen > 0:
# 			return self.frozenInstruction()
# 		return self.normalInstruction()
# 	def instructionToAction(self,index):
# 		return self.actions[index]
#
# class StochasticProgramCell(DiscreteProgramCell):
# 	# a program cell which represents the probabilities of actions
# 	def __init__(self,p,actions):
# 		maxInstr = len(p)
# 		DiscreteProgramCell.__init__(self,maxInstr,actions)
# 		self.p = p
# 	@overrides
# 	def frozenInstruction(self):
# 			self.timeUntilUnfrozen -= 1
# 			return np.argmax(self.p)
# 	@overrides
# 	def normalInstruction(self):
# 		polSum = 0
# 		r = random()
# 		for j in range(0, self.maxInstr):
# 			polSum += self.p[j]
# 			if (r <= polSum):
# 				return j
# 	def __getitem__(self, key):
# 		return self.p[key]
#
# 	def __setitem__(self,key,item):
# 		#print("setitem(cell)+" + str(self.p[key]))
# 		self.p[key] = item
# 		#print("setitem(cell)+" + str(self.p[key]))
# 	def __delitem__(self,key):
# 		del self.p[key]
# 	def __str__(self):
# 		return "StochasticProgramCell:"+str(self.p)
# 	__repr__ = __str__
#
# class ContinuousProgramCell(ProgramCell):
# 	# a program cell which represents the probabilities of actions
# 	def __init__(self, dimensions, parameters):
# 		self.dimensions=dimensions
# 		self.parameters=parameters
#
# 	def generateInstruction(self):
#          pass
#
#
# 	def __getitem__(self, key):
# 		return self.p[key]
#
# 	def __setitem__(self, key, item):
# 		# print("setitem(cell)+" + str(self.p[key]))
# 		self.p[key] = item
#
# 	# print("setitem(cell)+" + str(self.p[key]))
# 	def __delitem__(self, key):
# 		del self.p[key]
#
# 	def __str__(self):
# 		return "StochasticProgramCell:" + str(self.p)
#
# 	__repr__ = __str__
# class HierarchicalProgramCell(ProgramCell):
# 	# a program cell which is composed of different program cells
# 	def __init__(self, seq):
# 		self.seq = seq
#
#
# 	def __getitem__(self, key):
# 		return self.seq[key]
#
# 	def __setitem__(self, key, item):
# 		self.seq[key] = item
#
# 	def __delitem__(self, key):
# 		del self.seq[key]