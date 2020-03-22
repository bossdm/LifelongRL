
import numpy as np
from StatsAndVisualisation.LearnerStatistics import LearnerStatistics
from StatsAndVisualisation.metrics import getDifferential
from Actions.SpecialActions import PLAResult
from overrides import overrides
class SMP_Statistics(LearnerStatistics):
    def __init__(self,learner):
        self.instruction_frequencies={}
        self.instruction_frequenciesOverTime=[]
        self.numPModifications = 0
        self.changeSize = 0
        self.numPModificationsOverTime = []
        self.numPModificationsOverTime = []
        self.numValidPModificationsOverTime = []
        self.stackLengthOverTime = []
        self.changeSizeOverTime = []
        self.result=None
        LearnerStatistics.__init__(self,learner)
        self.init_action_freq(learner.actions)
    def init_action_freq(self,actions):
        for action in actions:
            self.instruction_frequencies[action.function.__name__]=0  # in case of duplication, just count them together
    def update_action_freq(self,action):
        self.instruction_frequencies[action.function.__name__] += 1
    def update(self,learner):
        if self.result is not None:
            self.numPModifications+=1
        self.result=None
        self.update_action_freq(learner.chosenAction)


    # def initialise_statistics(self):
    #     LearnerStatistics.initialise_statistics(self)
    #     self.developstats['Pmodifications'] = []
    #     self.developstats['validPmodifications'] = []
    #     self.finalstats['StackSize'] = 0


    # def development_stats(self,Vstep):
    #     Pmodifications.append(getDifferential(self.numPModificationsOverTime, j, Vstep))
    #     added_entries = getDifferential(self.numValidPModificationsOverTime, j, Vstep)
    #     validPmodifications.append(added_entries)
    #     PStackSize += added_entries
    #     StackSize += added_entries
    #     if method.startswith("SSA_NEAT") or method.startswith("SSA_WM_FixedNN") or method.startswith("SSA_FixedNN"):
    #         NPmodifications.append(getDifferential(self.numNPModificationsOverTime, j, Vstep))
    #         NPentries = getDifferential(self.numValidNPModificationsOverTime, j, Vstep)
    #         validNPmodifications.append(NPentries)
    #         NPStackSize += NPentries
    #         StackSize += NPentries
    #         modularity = self.modularityStats[j + Vstep]
    #         compress = self.compressStats[j + Vstep]
    #         numn = self.numNodesOverTime[j + Vstep]
    #         numc = self.numConnsOverTime[j + Vstep]
    #         for net in range(numnets):  # each network
    #             modularityStats[net].append(modularity[net])
    #             compressStats[net].append(compress[net])
    #             numnodes[net].append(numn[net])
    #             maxnodes = numnodes[net][-1]
    #             numconnections[net].append(numc[net])
class IS_LearnerStatistics(SMP_Statistics):

    def __init__(self,learner):
        SMP_Statistics.__init__(self,learner)
        self.numPModificationErrors=0
        self.numPModificationUnprepared=0
        self.numPModificationsRemoved=0
        self.numPModificationErrorsOverTime = []
    @overrides
    def update(self,learner):
        if self.result is not None:
            if self.result==PLAResult.success:
                self.numPModifications+=1
            elif self.result==PLAResult.error:
                self.numPModificationErrors+=1
            elif self.result==PLAResult.not_prepared:
                self.numPModificationUnprepared+=1
            elif self.result==PLAResult.removed_change:
                self.numPModificationsRemoved+=1
            else:
                pass
        self.result=None
        self.update_action_freq(learner.chosenAction)
    # def get_developstatskeys(self):
    #     return LearnerStatistics.get_developstatkeys(self)+['Pmodifications','validPmodifications','changeSize','stackLength']
    # def get_finalstatskeys(self):
    #     return LearnerStatistics.get_finalstatkeys(self)+['PStackSize','StackSize']
    def initialise_statistics(self):
        LearnerStatistics.initialise_statistics(self)
        self.developstats['Pmodifications']=[]
        self.developstats['validPmodifications']=[]
        self.developstats['changeSize'] = []
        self.developstats['stackLength'] = []
        self.finalstats['PStackSize']=0
        self.finalstats['StackSize']=0
        self.developstats['lr']=[]
        self.developstats['eps']=[]
        self.developstats['lrdeviation']=[]
        self.developstats['epsdeviation']=[]
        self.developstats['num_lr_changes']=[]
        self.developstats['num_eps_changes']=[]


    def development_statistics_iteration(self,j,Vstep,opt_speed=1.):


        LearnerStatistics.development_statistics_iteration(self,j,Vstep,opt_speed)
        if self.numValidPModificationsOverTime:
            #added_entries = getDifferential(self.numValidPModificationsOverTime, j, Vstep)
            added_entries=getDifferential(self.numValidPModificationsOverTime,j,Vstep)
            self.finalstats['PStackSize'] += added_entries
            self.finalstats['StackSize'] += added_entries
            self.developstats['validPmodifications'].append(added_entries)

        if hasattr(self,'changeSizeOverTime'):
            self.developstats['changeSize'].append(getDifferential(self.changeSizeOverTime, j, Vstep))
        if hasattr(self, 'stackLengthOverTime'):
            self.developstats['stackLength'].append(getDifferential(self.stackLengthOverTime, j, Vstep))

        if hasattr(self,"lr_overTime"):
            self.developstats['lr'].append(np.mean(self.lr_overTime[j:j + Vstep]))
            self.developstats['eps'].append(np.mean(self.eps_overTime[j:j + Vstep]))
            self.developstats['lrdeviation'].append(np.mean(self.lrdeviation_overTime[j:j + Vstep]))
            self.developstats['epsdeviation'].append(np.mean(self.epsdeviation_overTime[j:j + Vstep]))
            self.developstats['num_lr_changes'].append(sum(self.num_lr_changesOverTime[j:j + Vstep]))
            self.developstats['num_eps_changes'].append(sum(self.num_eps_changesOverTime[j:j + Vstep]))

class IS_NEAT_LearnerStatistics(IS_LearnerStatistics):
    def __init__(self,learner):
        IS_LearnerStatistics.__init__(self,learner)
        self.NPresult=None
        self.numNPModifications = 0
        self.numNPModificationErrors = 0
        self.numNPModificationUnprepared = 0
        self.numNPModificationsRemoved = 0
        self.numNPModificationsOverTime = []
        self.numNPModificationErrorsOverTime = []
        self.numValidNPModificationsOverTime = []
        self.numConnsOverTime = []
        self.numNodesOverTime = []
        self.modularityStats=[]
        self.compressStats=[]
        self.correctnessNP=[]
        self.totalNPusage = [] # number of uses of NP to output action

    # def update(self,learner):
    #
    #     IS_LearnerStatistics.update(self)
    # def get_developstatskeys(self):
    #     num_nets = len(self.compressStats[0])
    #     netkeys = ['NPusage%d' % (net) for net in range(num_nets)]
    #     netkeys += ['modularityStats%d' % (net) for net in range(num_nets)]
    #     netkeys += ['compressStats%d' % (net) for net in range(num_nets)]
    #     netkeys += ['numnodes%d' % (net) for net in range(num_nets)]
    #
    #
    #     return LearnerStatistics.get_developstatkeys(self)+['correctnessNP']+netkeys
    # @classmethod
    # def get_finalstatskeys(cls,numnets):
    #     num_nets = len(numnets)
    #     netkeys = ['finalNPusage%d' % (net) for net in range(num_nets)]
    #
    #     return LearnerStatistics.get_developstatkeys(self) + netkeys
    def initialise_statistics(self,numnets=1):
        IS_LearnerStatistics.initialise_statistics(self)
        self.finalstats['numnets'] = numnets
        self.finalstats['NPStackSize'] = 0
        for net in range(numnets):
            self.developstats['modularityStats%d'%(net)] = []
            self.developstats['compressStats%d'%(net)] = []
            self.developstats['numnodes%d'%(net)] = []
            self.developstats['numconnections%d'%(net)] = []
        self.developstats['correctnessNP']=self.correctnessNP
        self.developstats['NPmodifications']=self.numNPModificationsOverTime
        self.developstats['validNPmodifications']=[]

    def development_statistics(self,total_samples,Vstep,opt_speed,sample_rate=10000):

        IS_LearnerStatistics.development_statistics(self,total_samples,Vstep,opt_speed)
        total_time=total_samples*sample_rate
        # for net in range(self.finalstats['numnets']):  # each network
        #     self.developstats['NPusage%d' % (net)] = [self.totalNPusage[t][net] / float(total_time / len(self.totalNPusage))
        #                                             for t in range(len(self.totalNPusage))]
        #     self.finalstats['finalNPusage%d'%(net)] = self.totalNPusage[-1][net] / float(total_time / len(self.totalNPusage))

    def development_statistics_iteration(self,j,Vstep,opt_speed=1.):
        IS_LearnerStatistics.development_statistics_iteration(self,j,Vstep,opt_speed)
        if self.numValidNPModificationsOverTime:
            NPentries = getDifferential(self.numValidNPModificationsOverTime, j, Vstep)
        else:
            return
        self.finalstats['NPStackSize'] += NPentries
        self.finalstats['StackSize'] += NPentries
        self.developstats['validNPmodifications'].append(NPentries)
        modularity = self.modularityStats[j + Vstep]
        compress = self.compressStats[j + Vstep]
        numn = self.numNodesOverTime[j + Vstep]
        numc = self.numConnsOverTime[j + Vstep]
        for net in range(self.finalstats['numnets']):  # each network
            self.developstats['modularityStats%d'%(net)].append(modularity[net])
            self.developstats['compressStats%d'%(net)].append(compress[net])
            self.developstats['numnodes%d'%(net)].append(numn[net])
            #maxnodes = numnodes[net][-1]
            self.developstats['numconnections%d'%(net)].append(numc[net])



