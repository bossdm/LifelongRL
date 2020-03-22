
from argparse import ArgumentParser


def ParseBoolean (b):
    # ...
    if len(b) < 1:
        raise ValueError ('Cannot parse empty string into boolean.')
    # b = b[0].lower()
    if b == 'True':
        return True
    if b == 'False':
        return False
    raise ValueError ('Cannot parse string into boolean.')



parser = ArgumentParser()

# Add more options if you like in the experiment file
parser.add_argument('-t',dest="walltime",type=str)
parser.add_argument("-s",dest="STOPTIME",type=int)
parser.add_argument("-v", dest ="VISUAL",type=ParseBoolean)
parser.add_argument("-r", dest="run",type=int)
parser.add_argument("-m", dest="method") # self or trans or None
parser.add_argument("-f",dest="filename")
parser.add_argument("-e",dest="environment_file",type=ParseBoolean)
parser.add_argument("-c",dest="config_file",type=str)
parser.add_argument("-run_type",dest="run_type",type=str)
parser.add_argument("-record_video",dest="record_video",type=ParseBoolean)