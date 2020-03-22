
import dill as pickle
import os
import time
def printTotals(transferred, toBeTransferred):
    print("Transferred: {0}\tOut of: {1}".format(transferred, toBeTransferred))

def dump_incremental(filename,object):

    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    #data = bytearray(n_bytes)
    ## write
    print("starting pickle dump of %s (incremental)"%(filename))
    bytes_out = pickle.dumps(object)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])
    print("closed file")
    print("finished pickle dump of %s (incremental)"%(filename))


def dump_object(filename,e):
    with open(filename, "wb") as f:
        print("starting pickle dump")
        pickle.dump(e, f, pickle.HIGHEST_PROTOCOL)
        print("finished pickle dump")

def read_incremental(filename):
    print("loading file (incremental)")
    max_bytes=2**31 - 1
    ## read
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filename)
    with open(filename, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data2 = pickle.loads(bytes_in)
    print("file loaded successfully (incremental)")
    return data2
def getStatistics(e,filename):
    e.printStatistics()
    e.agent.learner.printPolicy()
    e.agent.learner.save_stats(filename)


def submit_job(arg_list,job_script,difficulty="Difficult"):
    arg_string =''
    found=False
    for i in range(len(arg_list)):
        if i > 0 and arg_list[i-1] == '-e':
            arg_string+='True'
            found=True
        else:
            arg_string+=arg_list[i]
        arg_string+=' '
    if not found:
        if difficulty=="Difficult":
            arg_string+='-e True -s 30000000'
        else:
            arg_string+='-e True'
    print(arg_string)
    os.system("export HOME=/home/db2c15")
    os.system(job_script+arg_string)

def get_record_intervals(stoptime):
    # total_video_time: is the total time in learner time steps

    record_slices = 10

    # with fps=10
    slice_time = 40# slice_time/10 seconds will be needed for one slice

    # convert into learner time steps (framerate=8.0)

    slice_space = stoptime/record_slices

    assert slice_time < slice_space
    interval_starts= [i* slice_space for i in range(record_slices)]
    return [[start, start + slice_time] for start in interval_starts]


def continue_experiment(interrupted,arg_list,job_script="bash ${HOME}/POmazescript.sh "):
    if interrupted:
        submit_job(arg_list,job_script)


def finalise_experiment(e,filename,arg_list,no_saving,args,save_stats=True,save_learner=True,save_environment=True):

    if no_saving:
        return

    # Saving the objects:
    if args.record_video:
        # can't save videowith pickle, so will have to stop this one
        del e.vis.display.video
        e.vis.display.on = False
    begintime = time.time()
    if not save_learner:
        e.agent.learner=None
    else:
        e.agent.learner.save(filename)
    if save_environment:
        dump_incremental(filename+"_environment", e)
    else:
        dump_incremental(filename+"_agent", e.agent)
    if save_stats:
        getStatistics(e, filename)


    time_passed = time.time() - begintime
    print("save time=%.3f" % (time_passed))


def save_intermediate(e,filename,save_stats=True,save_learner=True):
    if save_stats:
        getStatistics(e, filename)
    # Saving the objects:
    begintime = time.time()
    if not save_learner:
        e.agent.learner=None
    else:
        e.agent.learner.save(filename)
    dump_incremental(filename+"_environment", e)

    time_passed = time.time() - begintime
    print("save time=%.3f" % (time_passed))

