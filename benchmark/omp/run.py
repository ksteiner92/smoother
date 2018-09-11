import subprocess
import os
import numpy as np
import pandas as pd
from pandas import HDFStore
from astropy.stats import jackknife_stats

exe = "../../build/Release/improved"
in_file = "../../build/Release/Numeca-Logo.png"
out_file = "test.png"
out_hdf5 = "omp.hdf5"
nsteps = 10
r = 4

maxthreads = 32
minthreads = 1
nruns = 10

stime_res = np.empty(shape=(maxthreads - minthreads + 1, nruns, nsteps))
time_res = np.empty(shape=(maxthreads - minthreads + 1,nruns))

my_env = os.environ.copy()
irank = 0
ithread = 0
for nthreads in range(minthreads, maxthreads + 1):
    my_env["OMP_NUM_THREADS"] = str(nthreads)
    for i in range(0, nruns):
        print i + 1, "run with", nthreads, "threads"
        proc = subprocess.Popen([exe, in_file, out_file, str(nsteps), str(r)], env=my_env, stdout=subprocess.PIPE)
        output = proc.stdout.read()
        lines = output.splitlines()
        for line in lines:
            tokens = line.split()
            if not "done" in tokens:
                continue
            if "step" in tokens:
                step_pos = tokens.index("step")
                istep = int(tokens[step_pos + 1])
                time = float(tokens[step_pos + 3])
                stime_res[ithread,i,istep] = time
            else:
                step_pos = tokens.index("smoothing:")
                time = float(tokens[step_pos + 2])
                time_res[ithread,i] = time
    ithread += 1

tot_time = np.zeros((maxthreads - minthreads + 1, 2))
step_times = np.zeros((maxthreads - minthreads + 1, nsteps, 2))
step_tot_times = np.zeros((maxthreads - minthreads + 1,  2))
for i in range(maxthreads - minthreads + 1):
    tot_time[i, 0], bias, tot_time[i, 1], conf_interval = jackknife_stats(time_res[i, :], np.mean, 0.95)
    step_tot_times[i, 0], bias, step_tot_times[i, 1], conf_interval = jackknife_stats(stime_res[i, :, :].reshape((nsteps * nruns)), np.mean, 0.95)
    for istep in range(nsteps):
        step_times[i, istep, 0], bias, step_times[i, istep, 1], conf_interval = jackknife_stats(stime_res[i, :, istep], np.mean, 0.95)

DfTotTime = pd.DataFrame({"threads": np.arange(1, maxthreads - minthreads + 2), "time": tot_time[:,0], "error": tot_time[:,1]})
DfStepTimes = pd.DataFrame({"threads": np.arange(1, maxthreads - minthreads + 2), "time": step_times[:, 0, 0], "error": step_times[:, 0, 1]})
DfStepTimes["step"] = 0
for istep in range(1, nsteps):
    df = pd.DataFrame({"threads": np.arange(1, maxthreads - minthreads + 2), "time": step_times[:, istep, 0], "error": step_times[:, istep, 1]})
    df["step"] = istep
    DfStepTimes = pd.concat([DfStepTimes, df])

DfTotStepTimes = pd.DataFrame({"threads": np.arange(1, maxthreads - minthreads + 2), "time": step_tot_times[:,0], "error": step_tot_times[:,1]})

hdf = HDFStore(out_hdf5)
hdf.put("TotTimes",\
    DfTotTime,\
    format='table',\
    data_columns=True,\
    index=False)
hdf.put("StepTimes",\
    DfStepTimes,\
    format='table',\
    data_columns=True,\
    index=False)
hdf.put("TotStepTimes",\
    DfTotStepTimes,\
    format='table',\
    data_columns=True,\
    index=False)
hdf.close()
