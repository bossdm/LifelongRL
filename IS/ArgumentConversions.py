def narrowing_conversion(number, start,target):

    startMin,startMax=start
    targetMin,targetMax=target
    assert startMax - startMin >= targetMax - targetMin
    # num=int(targetMin + (number-startMin)*(targetMax-targetMin + .99)/(startMax-startMin))
    return int(targetMin + (number - startMin) * (targetMax - targetMin + .99) / (startMax - startMin))


def conversion(number, start, target):
    startMin, startMax = start
    targetMin, targetMax = target
    # num=int(targetMin + (number-startMin)*(targetMax-targetMin + .99)/(startMax-startMin))
    return int(targetMin + (number - startMin) * (targetMax - targetMin + .99) / (startMax - startMin))

def real_conversion(number, start, target):
    startMin, startMax = start
    targetMin, targetMax = target
    # num=int(targetMin + (number-startMin)*(targetMax-targetMin + .99)/(startMax-startMin))
    return targetMin + (number - startMin) * (targetMax - targetMin) / (startMax - startMin)