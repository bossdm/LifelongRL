from abc import abstractmethod
from overrides import overrides
import numpy as np
from random import random, randint
import copy
class ProgramCell(object):
	def __init__(self, maxInstr,time=None):
		self.timeUntilUnfrozen=0 if time==None else time
		self.maxInstr = maxInstr

	@abstractmethod
	def frozenInstruction(self):
		pass
	@abstractmethod
	def normalInstruction(self):
		pass
	def generateInstruction(self):
		if self.timeUntilUnfrozen > 0:
			return self.frozenInstruction()
		return self.normalInstruction()
class StochasticProgramCell(ProgramCell):
	def __init__(self,p,time=None):
		maxInstr = len(p)
		ProgramCell.__init__(self,maxInstr,time)
		self.p = p
	@classmethod
	def construct(cls,instance):
		return cls(copy.copy(instance.p),instance.timeUntilUnfrozen)
	@overrides
	def frozenInstruction(self):
			self.timeUntilUnfrozen -= 1
			return np.argmax(self.p)
	@overrides
	def normalInstruction(self):
		# polSum = 0
		# r = random()
		# for j in range(0, self.maxInstr):
		# 	polSum += self.p[j]
		# 	if (r <= polSum):
		# 		return j
		# return j #sometimes this happens due to rounding, even when dividing by probs by C ?
		return np.random.choice(self.maxInstr,  p=self.p)

	def __getitem__(self, key):
		return self.p[key]

	def __setitem__(self,key,item):
		#print("setitem(cell)+" + str(self.p[key]))
		self.p[key] = item
		#print("setitem(cell)+" + str(self.p[key]))
	def __delitem__(self,key):
		del self.p[key]
	def __str__(self):
		return "StochasticProgramCell:"+str(self.p)

	def __ne__(self, other):
		return not self == other
	def __eq__(self,other):
		return self.p == other.p and self.maxInstr == other.maxInstr and self.timeUntilUnfrozen == other.timeUntilUnfrozen
	def __hash__(self):
		return hash((tuple(self.p),self.maxInstr,self.timeUntilUnfrozen))
	def __deepcopy__(self, memodict={}):
		# new_one=StochasticProgramCell(copy.deepcopy(self.p),self.timeUntilUnfrozen)
		# return new_one
		return StochasticProgramCell.construct(self)
	__repr__ = __str__
class HierarchicalProgramCell(ProgramCell):
	# a program cell which is composed of different program cells
	def __init__(self, seq):
		maxInstr = len(seq)
		ProgramCell.__init__(self, maxInstr)
		self.seq = seq

	@overrides
	def frozenInstruction(self):
		pass

	@overrides
	def normalInstruction(self):
		pass

	def __getitem__(self, key):
		return self.seq[key]

	def __setitem__(self, key, item):
		self.seq[key] = item

	def __delitem__(self, key):
		del self.seq[key]
class Policy(object):
	def __init__(self,pol,programCellTypes):
		self.p=[]
		for i in range(len(pol)):
			self.p[i] = programCellTypes[i](pol[i])

	def generateInstruction(self,IP):
		return self.p[IP].generateInstruction()
	def __getitem__(self, key):
		return self.p[key]
	def __setitem__(self,key,item):
		#print("setitem+" + str(self.p[key]))
		self.p[key] = item
		#print("setitem+" + str(self.p[key]))
	def __delitem__(self,key):
		del self.p[key]
	def asMatrix(self):
		return [vec for vec in self.p]

	@classmethod
	def from_matrix(cls,matrix):
		P=cls(matrix)


class MapPolicy(object):
	def __init__(self,pol,programCellTypes,times=None):
		self.p={}
		if times is None:
			times=[None]*len(programCellTypes)
		allTheSame=False
		if len(programCellTypes) == 1:
			allTheSame=True
		j=0
		for i in pol:
			self.p[i] = programCellTypes[j](pol[i],times[j])
			if not allTheSame:
				j+=1
	# def from_program_cells(cls,instance):
	# 	new_p={}
	# 	for IP, p in instance.items():
	# 		new_p[IP]=copy.deepcopy(p)


	def generateInstruction(self,IP):
		#print(self.p[IP])
		return self.p[IP].generateInstruction()
	def asMatrix(self):
		mat=[row.p for row in self.p.values()]
		return np.array(mat)
	def __getitem__(self, key):
		return self.p[key]
	def __setitem__(self,key,item):
		#print("setitem+" + str(self.p[key]))
		self.p[key] = item
		#print("setitem+" + str(self.p[key]))
	def __delitem__(self,key):
		del self.p[key]
	def __deepcopy__(self, memodict={}):
		cells=self.p.values()
		ps={IP:copy.deepcopy(p.p) for (IP, p) in self.p.items()}
		types=[type(p) for p in cells]
		times=[p.timeUntilUnfrozen for p in cells]
		new_one=MapPolicy(ps,types,times )
		return new_one


class MultiPolicy(object):
	def __init__(self, pols):
		self.pols=pols
	def generateInstruction(self, PolP, IP):
		return self.pols[PolP].generateInstruction(IP)

	def __getitem__(self, key):
		return self.pols[key]

	def __setitem__(self, key, item):
		# print("setitem+" + str(self.p[key]))
		self.pols[key] = item

	# print("setitem+" + str(self.p[key]))
	def __delitem__(self, key):
		del self.pols[key]