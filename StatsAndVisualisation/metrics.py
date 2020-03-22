print_path=False
import numpy as np

def getDifferential(array, begin, step):
    print(begin)
    print(step)
    if isinstance(array[begin+step],list):
        l1 = array[begin+step]
        l2 = array[begin]
        return [l1[index] - l2[index] for index in range(len(l1))]
    return array[begin+step] - array[begin]



# def F_test(data,type="s"):
#     # s: standard F-test
#     # t: trimmed mean
#     # b: BrownForsythe_test; is an ANOVA test suitable for asymmetric distributions e.g. Chi-Squared
#     # assuming first dimension is the group index
#     num_groups=len(data)
#     num_observations = [len(data[i]) for i in range(num_groups)]
#     total_observations = sum(num_observations)
#     #transform data
#     z=[]
#     grand_mean=0
#     for i in range(num_groups):
#         med=np.median(data[i])
#         z.append([0 for k in range(num_observations[i])])
#         for j in range(num_observations[i]):
#             if type == "b":
#                 z[i][j]=abs(data[i][j]-med)
#             elif type == "s":
#                 z[i][j]=data[i][j]
#             grand_mean+=z[i][j]
#     grand_mean/=total_observations
#
#     between = sum([num_observations[i]*(np.mean(z[i])-grand_mean)**2])
#     within = sum([sum([(z[i][j]-np.mean(z[i]))**2 for j in range(num_observations[i])]) for i in range(num_groups)])
#
#     F= (total_observations - num_groups)/float(num_groups - 1)*between/float(within)
#     from scipy.stats import f
#     dfn = num_groups - 1
#     dfd = total_observations - num_groups
#     p = 1 - f.cdf(F, dfn, dfd, loc=0, scale=1)
#     return F, p

def pathlengths(stats,slices,optimum):
    # problem specifics
    bps = stats.problem_specific['best_path']

    wps = stats.problem_specific['worst_path']

    sdps = stats.problem_specific['sd_path']

    gs = []
    try:
        gs = stats.problem_specific['goals']
    except:
        if print_path:
            print("WARNING: did not record goal achievements")
    mps = []
    try:
        mps = stats.problem_specific['mean_path']
    except:
        if print_path:
            print("WARNING: did not record mean paths")
    bestpaths = []
    worstpaths = []
    meanpaths = []
    sdpaths = []

    for t in range(slices):
        numcoords = 0
        sumweight = 0
        sdpath = 0
        meanpath = 0
        bestpath = 100000
        worstpath = 1
        bps[t] = bps[t][0] if isinstance(bps[t], list) else bps[t]  # made list instead (error)
        wps[t] = wps[t][0] if isinstance(wps[t], list) else wps[t]  # made list instead (error)
        if mps:
            mps[t] = mps[t][0] if isinstance(mps[t], list) else mps[t]  # made list instead (error)
        sdps[t] = sdps[t][0] if isinstance(sdps[t], list) else sdps[t]  # made list instead (error)
        for coord, path in bps[t].items():
            if print_path:
                print("start-goal = " + str(coord))
                print("best path = " + str(path[0]))
                print("worst path =" + str(wps[t][coord][0]))
                print("sd path =" + str(sdps[t][coord]))
                if gs:
                    print("goals achieved: " + str(gs[t][coord]))
                if mps:
                    print("mean path: " + str(mps[t][coord]))
            bp = path[0] / float(optimum)
            wp = wps[t][coord][0] / float(optimum)
            if bp < bestpath:
                bestpath = bp
            if wp > worstpath:
                worstpath = wp
            if not gs:
                sdpath += sdps[t][coord] / float(optimum)
                if mps:
                    meanpath += mps[t][coord] / float(optimum)
            else:
                # assign weight based on gs
                weight = gs[t][coord]
                sdpath += weight * sdps[t][coord] / float(optimum)
                if mps:
                    meanpath += weight * mps[t][coord] / float(optimum)
                sumweight += weight

            numcoords += 1
        if gs:
            if sumweight > 0:
                sdpath /= sumweight
            else:
                sdpath = 0  # no goal achievements
        else:
            sdpath /= numcoords

        bestpaths.append(bestpath)
        worstpaths.append(worstpath)
        sdpaths.append(sdpath)

        if mps:
            meanpaths.append(meanpath)
    return bestpaths,worstpaths,sdpaths,meanpaths
