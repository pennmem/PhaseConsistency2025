import base64
import datetime
import functools
import io
import json
import math
import numpy as np
import os
import pandas as pd
import pylab as plt
import scipy
import scipy.stats
import statsmodels.stats.multitest
import statsmodels.stats.weightstats
import sys
import time
import traceback
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import morlet
from ptsa.data.filters import ButterworthFilter
from Locator import Locator


def SetLogger(logfile='logfile.txt', suffix='', sep=', ', datestamp=True,
    stdout=True):
  '''Preconfigure the defaults to a logging function returned by this.'''
  def LogErr(*args, logfile=logfile, suffix=suffix, sep=sep,
      datestamp=datestamp, stdout=stdout, **kwargs):
    '''Error logging function suitable for single process or parallel use.
       All parameters other than ones specified below are formatted with a
       preceding date/time stamp, and then printed to output as well as
       appended to the logfile.  Extra named parameters are printed along
       with their names as labels.  Any exceptions passed in are formatted
       on the main output line, and then the traceback is appended in
       subsequent lines to the output.
       sep: The separator between printable outputs (default: ', ').
       logfile: The starting filename for the output log file
                (default: 'logfile.txt').
       suffix: For example, with logfile='logfile.txt', makes the new
               logfile be 'logfile_'+str(suffix)+'.txt'.  Use this to label
               log files by parameter.
       datestamp: If True, the date/time stamp is prepended to the line.
       stdout: If True, output is echoed to stdout.'''
    import datetime
    import traceback
    import os

    def UnicodeClean(s):
      '''Guarantees string s will encode as valid utf-8, with best-effort
         re-interpretation of invalid surrogate data.'''
      try:
        s.encode('utf-8')
        return s
      except Exception:
        pass
      try:
        s = s.encode('utf-16', 'surrogatepass').decode('utf-16')
        return s
      except Exception:
        pass
      try:
        s = s.encode('ascii', 'surrogateescape').decode('iso-8859-1')
        return s
      except Exception:
        pass
      s = s.encode('utf-8', 'ignore').decode('utf-8')
      return s

    arglist = []
    exclist = []
    for a in args:
      if isinstance(a, BaseException):
        arglist.append(traceback.format_exception_only(type(a), a)[0].strip())
        exclist.append(a)
      else:
        arglist.append(a)

    if datestamp:
      s = datetime.datetime.now().strftime('%F_%H-%M-%S') + ': '
    else:
      s = ''
    s += sep.join(str(a) for a in arglist)
    if arglist and kwargs:
      s += sep
    s += sep.join(str(k)+'='+str(v) for k,v in kwargs.items())
    for e in exclist:
      s += '\n' + \
          ''.join(traceback.format_exception(type(e), e, e.__traceback__))

    logfile,ext = os.path.splitext(logfile)
    suffix = str(suffix)
    if suffix:
      logfile += '_'+suffix
    logfile += ext

    if stdout:
      print(s)
    with open(logfile, 'a') as fw:
      fw.write(UnicodeClean(s+'\n'))

  return LogErr

LogErr = SetLogger()


def DFRtoDict(df_row):
  '''Convenience function for turning a DataFrame row into a dict.'''
  try:  # Check if dict-like
    dr = dict(df_row)
  except Exception as e:
    try:
      dr = df_row._asdict() # Try for pandas.core.frame.Pandas
    except AttributeError:
      dr = df_row.to_dict() # Try for pandas.core.series.Series
  return dr


def Log(s, suffix=''):
  '''Deprecated.  Use LogErr.'''
  date = datetime.datetime.now().strftime('%F_%H-%M-%S')
  output = date + ': ' + str(s)

  filename = 'analysis_log'
  suffix = str(suffix)
  if suffix != '':
    filename = filename + '_' + suffix
  filename = filename + '.txt'

  with open(filename, 'a') as logfile:
    print(output)
    logfile.write(output+'\n')


def LogException(e, suffix=''):
  '''Deprecated.  Use LogErr.'''
  Log(e.__class__.__name__+', '+str(e)+'\n'+
      ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
      suffix = suffix)


def LogDFErr(row, *args, LogErr=LogErr, **kwargs):
  rd = DFRtoDict(row)
  LogErr(*args, sub=str(rd['subject']), exp=str(rd['experiment']),
      sess=str(rd['session']), **kwargs)

def LogDFException(row, e, *args, **kwargs):
  '''Deprecated.  Use LogDFErr'''
  LogDFErr(row, e, *args, **kwargs)


# JSSave/JSLoad bundle
def DataPack(x):
  import base64
  import numpy as np
  import pandas as pd
  try:
    if isinstance(x, np.ndarray):
      if x.dtype == 'O':
        raise TypeError('Unsupported')
      '''Converts a numpy array to a serializable string.'''
      b64 = base64.encodebytes(x.tobytes()).strip().decode()
      shape_str = ','.join(str(s) for s in x.shape)
      return f'@array;{x.dtype.str};{shape_str};{b64}'
    elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
      '''Converts a pandas DataFrame or Series to a serializable string.'''
      s = x.to_csv(index=None)
      return f'@table;{s}'
  except Exception:
    pass
  raise TypeError('Unsupported')

def DataUnpack(s):
  import base64
  import numpy as np
  import pandas as pd
  import io
  try:
    if isinstance(s, str):
      parts = s.split(';')
      if len(parts) == 4 and parts[0] == '@array':
        '''Unpacks a numpy array from a serializable string.'''
        dtype, shape_str, b64 = parts[1:]
        buf = base64.decodebytes(b64.encode())
        shape = [int(e) for e in shape_str.split(',') if e]
        return np.frombuffer(buf, dtype=dtype).reshape(shape)
      elif len(parts) >= 2 and parts[0] == '@table':
        '''Unpacks a pandas DataFrame or Series from a serializable string.'''
        s = ';'.join(parts[1:])
        df = pd.read_csv(io.StringIO(s))
        if len(df.columns) == 1:
          # Convert to series
          df = df[df.columns[0]]
        return df
  except Exception:
    pass
  return s

def TraverseUnpack(d):
  if isinstance(d, dict):
    return {k:TraverseUnpack(v) for k,v in d.items()}
  elif isinstance(d, list):
    return [TraverseUnpack(e) for e in d]
  elif isinstance(d, str):
    return DataUnpack(d)
  else:
    return d

def JSLoad(filename):
  import json
  with open(filename, 'r') as fr:
    data = json.load(fr)
  return TraverseUnpack(data)

def JSLoadStr(s):
  import json
  data = json.loads(s)
  return TraverseUnpack(data)

def JSSave(filename, data):
  import json
  with open(filename, 'w') as fw:
    json.dump(data, fw, indent=2, default=DataPack)

def JSSaveStr(data):
  import json
  return json.dumps(data, indent=2, default=DataPack)
# End JSSave/JSLoad bundle


class Settings():
  '''settings = Settings()
     settings.somelist = [1, 2, 3]
     settings.importantstring = 'saveme'
     settings.Save()

     settings = Settings.Load()
  '''
  def __init__(self, **kwargs):
    for k,v in kwargs.items():
      self.__dict__[k] = v

  def Save(self, filename='settings.json'):
    JSSave(filename, self.__dict__)

  def Load(filename='settings.json'):
    return Settings(**JSLoad(filename))

  def Store(self, filename, **kwargs):
    '''Store extra info in self.outdir subdirectory.'''
    import os
    if not filename.endswith('.json'):
      filename = f'{filename}.json'
    if hasattr(self, 'outdir'):
      filename = os.path.join(self.outdir, filename)
    Settings(**kwargs).Save(filename)

  def Grab(self, filename):
    '''Retrieve extra info from self.outdir subdirectory.'''
    import os
    if not filename.endswith('.json'):
      filename = f'{filename}.json'
    if hasattr(self, 'outdir'):
      filename = os.path.join(self.outdir, filename)
    return Settings.Load(filename)

  def __repr__(self):
    return ('Settings(' +
      ', '.join(str(k)+'='+repr(v) for k,v in self.__dict__.items()) +
      ')')

  def __str__(self):
    return '\n'.join(str(k)+': '+str(v) for k,v in self.__dict__.items())


class RunOpts(Settings):
  '''A Settings subclass which parses command line options to select which
     portions of a program to run.
     Usage:  run = RunOpts(__name__, ['setup', 'calculate', 'plots'])
             if run.setup:
               DoSetupThings()
     namevar should be __name__ in the main file.
     parts should be a list of parts to be set True if none are passed on the
     command line, or if some are, only the ones passed are True.
  '''
  def __init__(self, namevar, parts, **kwargs):
    self.main = namevar == '__main__'
    if self.main:
      cmd_params = sys.argv[1:]
      for c in cmd_params:
        if c not in parts:
          print('Options:', ' '.join(parts))
          sys.exit(-1)
      if cmd_params:
        for p in parts:
          self.__dict__[p] = p in cmd_params
      else:
        for p in parts:
          self.__dict__[p] = True
    else:
      for p in parts:
        self.__dict__[p] = False

    super().__init__(**kwargs)


class Benchmark():
  def __init__(self):
    self.times = {}
    self.locs = {}
    self.initial_time = time.perf_counter()

  def Start(self, label=None):
    timelist = self.times.setdefault(label, [])
    if (len(timelist)>0) and (timelist[-1][1]) is None:
      raise ValueError('Double Benchmark start for "'+str(label)+'"')
    timelist.append([time.perf_counter(), None, 0])

  def Stop(self, label=None, cnt=1):
    timelist = self.times[label]
    if timelist[-1][1] is not None:
      raise ValueError('Double Benchmark stop for "'+str(label)+'"')
    timelist[-1][1] = time.perf_counter()
    timelist[-1][2] = cnt

  def Total(self):
    return time.perf_counter() - self.initial_time

  def TotalStr(self):
    total = time.perf_counter() - self.initial_time
    sparts = []
    if total > 60*60*24:
      days = total // (60*60*24)
      total -= days * 60*60*24
      label = 'days' if days > 1 else 'day'
      sparts.append(f'{days:0.0f} {label}')
    if total > 60*60:
      hours = total // (60*60)
      total -= hours * 60*60
      sparts.append(f'{hours:0.0f} hr')
    if total > 60:
      minutes = total // 60
      total -= minutes * 60
      sparts.append(f'{minutes:0.0f} min')
    sparts.append(f'{total:0.3f} s')
    return ', '.join(sparts)

  def Report(self):
    report = 'label, count, avg_time\n'
    for label,timelist in self.times.items():
      starts = [t[0] for t in timelist]
      stops = [t[1] for t in timelist]
      cnts = [t[2] for t in timelist]
      cntsum = sum(cnts)
      if None in stops:
        raise ValueError('Unstopped Benchmark start for "'+str(label)+'"')
      diff = sum([e-b for b,e in zip(starts, stops)])
      report += str(label) + ', '
      report += str(cntsum) + ', '
      report += str(diff/cntsum) + '\n'
    return report

  def SaveReport(self, filename):
    with open(filename, 'w') as fw:
      fw.write(self.Report())

bench = Benchmark()


def StartFig(*args, **kwargs):
  fig = plt.figure(*args, **kwargs)
  plt.rcParams.update({'font.size': 12})
  return fig


def SaveFig(basename, setdir=None, store_setdir=['.']):
  if setdir is not None:
    store_setdir[0] = setdir

  if basename is None:
    return

  if os.sep in basename:
    pathname = basename
  else:
    pathname = os.path.join(store_setdir[0], basename)

  plt.savefig(pathname+'.png')
  plt.savefig(pathname+'.pdf')
  plt.show()
  plt.close()


def ConfidenceIntervals(a, conf=0.95, axis=0):
  N = a.shape[axis]
  barfactor = scipy.stats.t.interval(conf, N)[1]
  sems = scipy.stats.sem(a, axis=axis)
  return sems*barfactor


def RhinoRoot(path=None, _stored_here=[None]):
  if path is not None:
    _stored_here[0] = path
  return _stored_here[0]


def ExpTypes(search_str=''):
  '''Returns a list of all experiment types containing the search string.

     The rhino root directory can be set with RhinoRoot().
  '''
  df = get_data_index('all', rootdir=RhinoRoot())
  exp_types = set(df['experiment'])
  exp_list = sorted(exp for exp in exp_types if search_str in exp)
  return exp_list


def DataFramesFor(exp_list):
  '''Returns all dataframes for a list of experiments.

     The rhino root directory can be set with RhinoRoot().
  '''
  if isinstance(exp_list, str):
    exp_list = [exp_list]

  df = get_data_index('all', rootdir=RhinoRoot())
  indices_list = [df['experiment']==exp for exp in exp_list]
  indices = functools.reduce(lambda x,y: x|y, indices_list)
  df_matched = df[indices]
  return df_matched


def CMLReadDFRow(row):
  '''for row in df.itertuples():
          reader = CMLReadDFRow(row)

     The rhino root directory can be set with RhinoRoot().
  '''
  try:
    rd = row._asdict() # Try for pandas.core.frame.Pandas
  except AttributeError:
    rd = row.to_dict() # Try for pandas.core.series.Series
  return CMLReader(rd['subject'], rd['experiment'], rd['session'], \
                   rd['montage'], rd['localization'], rootdir=RhinoRoot())


# Uses ind.region
def MakeLocationFilter(scheme, location):
  return [location in s for s in [s if s else '' for s in scheme.iloc()[:]['ind.region']]]


def SubjectDataFrames(sub_list, exp_list_filter=None):
  '''Return all dataframes for a subject or list of subjects, if the
     experiment matches one in exp_list_filter (or all if None).

     The rhino root directory can be set with RhinoRoot().
  '''
  if isinstance(sub_list, str):
      sub_list = [sub_list]

  df = get_data_index('all', rootdir=RhinoRoot())
  indices_list = [df['subject']==sub for sub in sub_list]
  indices = functools.reduce(lambda x,y: x|y, indices_list)
  df_matched = df[indices]

  if exp_list_filter is not None:
    indices_list = [df_matched['experiment']==exp for exp in exp_list_filter]
    indices = functools.reduce(lambda x,y: x|y, indices_list)
    df_matched = df_matched[indices]

  return df_matched


def GetElectrodes(sub):
  df_sub = SubjectDataFrames(sub)
  reader = CMLReadDFRow(next(df_sub.itertuples()))
  # For scalp data, this is currently only accesible via ptsa.
  # So this is the most general method for all data.
  evs = reader.load('events')
  enc_evs = evs[evs.type=='WORD']
  eeg = reader.load_eeg(events=evs, rel_start=0, rel_stop=500, clean=True)
  return eeg.to_ptsa().channel.values


def CircularDispersion(arr):
  '''Calculate the circular dispersion for a 1D array of complex values.
     From Fisher 1993, Statistical Analysis of Circular Data, eq. 2.28.'''
  Cbar = np.mean(np.real(arr))
  Sbar = np.mean(np.imag(arr))
  Rbar = np.sqrt(Cbar*Cbar+Sbar*Sbar)
  C2bar = np.mean(np.cos(2*np.angle(arr)))
  S2bar = np.mean(np.sin(2*np.angle(arr)))
  rho2 = np.sqrt(C2bar*C2bar+S2bar*S2bar)
  return (1-rho2)/(2*Rbar*Rbar)


def CircularConfidence(circ_disp, N):
  '''95% confidence interval of phase, in radians, plus or minus the
     value returned.  From Fisher 1993, Statistical Analysis of Circular
     Data, eqns. 2.28 and 4.22.'''
  z_sigma = 1.959963984540054235*circ_disp/N
  if (z_sigma >= 1):
    return np.pi/2
  return np.arcsin(z_sigma)


def PhaseSpread(circ_disp, N):
  '''One sigma spread of phase, in radians.  Adapted from Fisher 1993,
     Statistical Analysis of Circular Data, eqns. 2.28 and 4.22, and
     modified to produce one sigma spread.  Verified by simulation,
     but note this picks up a sample size dependence at high spread.'''
  scaled_sigma = np.sqrt(circ_disp/(N-1))
  if scaled_sigma >= 1:
    return np.pi/2
  retval = np.arcsin(scaled_sigma)*np.sqrt(N)
  if retval > np.pi/2:
    return np.pi/2
  return retval


def PhaseConsistency(arr, N):
  '''Returns a shifted degree-of-freedom scaled Rayleigh z-score with average
     values ranging from 0 to 1, and with individual returned values with noise
     ranging 1 or less.  0 is the expectation value for no consistency in
     phase, while 1 is the value for perfect consistency in phase.  The
     averages of these phase consistency values are verified by simulation to
     be invariant across sample sizes of N>=2.  Adapted from Zar eq. 27.2'''
  Cbar = np.mean(np.real(arr))
  Sbar = np.mean(np.imag(arr))
  r_sqrd = Cbar*Cbar+Sbar*Sbar
  z = N*r_sqrd
  zs = (z-1)/(N-1)
  return zs


def Rebin(a, axis, binby, strict=False):
  '''Rebin a numpy array on the given axis by nearest-neighbor averaging,
     binning it down by integer factor binby.  If there are extra elements in
     the axis which don't bin evenly, they are discarded if strict=False, or
     ValueError is raised if strict=True.'''
  if not isinstance(a, np.ndarray):
    a = np.array(a)

  orig_binby = binby
  binby = int(binby)
  if binby != orig_binby:
    raise ValueError('Must rebin by an integer factor.  Got ' + \
      str(orig_binby))

  newdim_flt = a.shape[axis] / binby
  newdim = int(newdim_flt)
  if newdim != newdim_flt:
    if strict:
      raise ValueError(\
        'strict=True so axis size must be divisible by binby.  For ' + \
        str(a.shape[axis]) + '/' + str(binby))
    else:
      a = a.take(np.arange(0,newdim*binby), axis=axis)

  newshape = (*a.shape[:axis], newdim, binby, *a.shape[axis+1:])
  return a.reshape(newshape).mean(axis=axis+1)


def WeightedMeanSDSem(a, Ns):
  '''For 1D or 2D a, NS of same size, returns mu, SD, sem of a, all
     weighted by Ns.'''
  norm_w_a = np.array(Ns)/np.mean(Ns)

  desc = statsmodels.stats.weightstats.DescrStatsW(a, norm_w_a, ddof=1)
  return desc.mean, desc.std, desc.std_mean


def WeightedN1SampTTest(a, mu, Ns):
  '''Perform a 1 sample t-test weighted by the number of contributers to
     each.  Returns t, p.'''
  a = np.array(a)
  norm_w_a = np.array(Ns)/np.mean(Ns)

  desc = statsmodels.stats.weightstats.DescrStatsW(a, norm_w_a, ddof=1)
  t, p, df = desc.ttest_mean(mu)
  return t, p


def WeightedNPairedTTest(a, b, Ns):
  '''Perform a paired t-test weighted by the number of contributers to
     each.  Returns t, p.'''
  d = np.array(a)-np.array(b)
  norm_w_d = np.array(Ns)/np.mean(Ns)

  desc = statsmodels.stats.weightstats.DescrStatsW(d, norm_w_d, ddof=1)
  t, p, df = desc.ttest_mean(0)
  return t, p


def WeightedVarPairedTTest(a, b, var_a, var_b):
  '''Perform a paired t-test weighted by the inverse of the combined
     variances of a and b.  Returns t, p.'''
  d = np.array(a)-np.array(b)
  var_d = np.array(var_a) + np.array(var_b)
  weights_d = 1/var_d
  norm_w_d = weights_d/np.mean(weights_d)

  desc = statsmodels.stats.weightstats.DescrStatsW(d, norm_w_d, ddof=1)
  t, p, df = desc.ttest_mean(0)
  return t, p


def AUCFromH5(classifier_h5_file):
  from sklearn.metrics import roc_auc_score
  import h5py

  with h5py.highlevel.File(classifier_h5_file, 'r') as hf:
    return roc_auc_score(hf['_true_outcomes'][()],
                         hf['_predicted_probabilities'][()])


def RecallSpos(rec_event, enc_evs):
  '''rec_event must be a successfully recalled event, i.e. where
     rec_evs['recalled']==1'''
  event = enc_evs[(enc_evs['subject']==rec_event.subject) &
                  (enc_evs['session']==rec_event.session) &
                  (enc_evs['list']==rec_event.list) &
                  (enc_evs['item_num']==rec_event.item_num)]
  spos = event['serialpos'].values[0]
  return spos


class SubjectStats():
  def __init__(self):
    self.sessions = 0
    self.lists = []
    self.recalled = []
    self.intrusions_prior = []
    self.intrusions_extra = []
    self.repeats = []
    self.num_words_presented = []

  def Add(self, evs):
    enc_evs = evs[evs.type=='WORD']
    rec_evs = evs[evs.type=='REC_WORD']

    # Trigger exceptions before data collection happens
    enc_evs.recalled
    enc_evs.intrusion
    enc_evs.item_name
    if 'trial' in enc_evs.columns:
        enc_evs.trial
    else:
        enc_evs.list

    self.sessions += 1
    if 'trial' in enc_evs.columns:
        self.lists.append(len(enc_evs.trial.unique()))
    else:
        self.lists.append(len(enc_evs.list.unique()))
    self.recalled.append(sum(enc_evs.recalled))
    self.intrusions_prior.append(sum(rec_evs.intrusion > 0))
    self.intrusions_extra.append(sum(rec_evs.intrusion < 0))
    words = rec_evs.item_name
    self.repeats.append(len(words) - len(words.unique()))
    self.num_words_presented.append(len(enc_evs.item_name))

  def ListAvg(self, arr):
    return np.sum(arr)/np.sum(self.lists) if np.sum(self.lists)>0 else math.nan

  def RecallFraction(self):
    return np.sum(self.recalled)/np.sum(self.num_words_presented)


#def SubjectStatTable(subjects, subfilt=None):
def SubjectStatTable(dfs, subfilt=None):
  ''' Prepare LaTeX table of subject stats '''
  table = ''
  try:
    table += '\\begin{tabular}{lrrrrrr}\n'
    table += ' & '.join('\\textbf{{{0}}}'.format(h) for h in [
        'Subject',
        '\\# Sessions',
        '\\# Lists',
        'Recalled Fraction',
        #'Avg Recalled',
        'Prior Intrusions',
        'Extra Intrusions',
        'Repeats'
    ])
    table += ' \\\\\n'

    for sub,df_sub in dfs.groupby('subject'):
      try:
        stats = SubjectStats()
        for row in df_sub.itertuples():
          reader = CMLReadDFRow(row)
          evs = reader.load('task_events')
          stats.Add(evs)
      except Exception as ex:
        LogDFErr(df_sub, ex)
        continue
      include = True
      if subfilt:
        include = subfilt(stats)
      if include:
        table += ' & '.join([sub, str(stats.sessions)] +
          ['{:.2f}'.format(x) for x in [
            np.sum(stats.lists),
            #stats.ListAvg(stats.recalled),
            stats.RecallFraction(),
            stats.ListAvg(stats.intrusions_prior),
            stats.ListAvg(stats.intrusions_extra),
            stats.ListAvg(stats.repeats)
          ]]) + ' \\\\\n'

    table += '\\end{tabular}\n'
  except Exception as e:
      print (table)
      raise

  return table


def ClusterRunSGE(function, parameter_list, max_jobs=100, procs_per_job=1):
  '''function: The routine run in parallel, which must contain all necessary
     imports internally.

     parameter_list: should be an iterable of elements, for which each
     element will be passed as the parameter to function for each parallel
     execution.

     max_jobs: Standard Rhino cluster etiquette is to stay within 100 jobs
     running at a time.  Please ask for permission before using more.

     procs_per_job: The number of concurrent processes to reserve per job.

     In jupyterlab, the number of engines reported as initially running may
     be smaller than the number actually running.  Check usage from an ssh
     terminal using:  qstat -f | egrep "$USER|node" | less

     Undesired running jobs can be killed by reading the JOBID at the left
     of that qstat command, then doing:  qdel JOBID
  '''
  import cmldask.CMLDask as da
  from dask.distributed import wait, as_completed, progress

  num_jobs = len(parameter_list)
  num_jobs = min(num_jobs, max_jobs)

  with da.new_dask_client_sge(function.__name__, '5GB', max_n_jobs=num_jobs,
      processes_per_job=procs_per_job) as client:
    futures = client.map(function, parameter_list)
    wait(futures)
    res = client.gather(futures)

  return res


def ClusterRunSlurm(function, parameter_list, max_jobs=64, procs_per_job=1,
    mem='5GB'):
  '''function: The routine run in parallel, which must contain all necessary
     imports internally.

     parameter_list: should be an iterable of elements, for which each
     element will be passed as the parameter to function for each parallel
     execution.

     max_jobs: Standard Rhino cluster etiquette is to stay within 100 jobs
     running at a time.  Please ask for permission before using more.

     procs_per_job: The number of concurrent processes to reserve per job.

     mem: A string specifying the amount of RAM required per job, formatted
     like '5GB'.  Standard Rhino cluster etiquette is to stay within 320GB
     total across all jobs.

     In jupyterlab, the number of engines reported as initially running may
     be smaller than the number actually running.  Check usage from an ssh
     terminal using:  "squeue" or "squeue -u $USER"

     Undesired running jobs can be killed by reading the JOBID at the left
     of that squeue command, then doing:  scancel JOBID
  '''
  import cmldask.CMLDask as da
  from dask.distributed import wait, as_completed, progress

  num_jobs = len(parameter_list)
  num_jobs = min(num_jobs, max_jobs)

  with da.new_dask_client_slurm(function.__name__, mem, max_n_jobs=num_jobs,
      processes_per_job=procs_per_job) as client:
    futures = client.map(function, parameter_list)
    wait(futures)
    res = client.gather(futures)

  return res


def ClusterRun(*args, **kwargs):
  mapping = {'sge':ClusterRunSGE, 'slurm':ClusterRunSlurm}
  dispatch = mapping['slurm']
  max_jobs = 100
  procs_per_job = 1
  mem = '5GB'

  if 'settings' in kwargs:
    dispatch = mapping[kwargs['settings'].scheduler]
    if 'max_jobs' in kwargs['settings'].__dict__:
      max_jobs = kwargs['settings'].max_jobs
    if 'procs_per_job' in kwargs['settings'].__dict__:
      procs_per_job = kwargs['settings'].procs_per_job
    if 'mem' in kwargs['settings'].__dict__:
      mem = kwargs['settings'].mem
    kwargs.pop('settings')

  if len(args) > 2:
    max_jobs = args[2]
  elif 'max_jobs' in kwargs:
    max_jobs = kwargs['max_jobs']
    kwargs.pop('max_jobs')
  if len(args) > 3:
    procs_per_job = args[3]
  elif 'procs_per_job' in kwargs:
    procs_per_job = kwargs['procs_per_job']
    kwargs.pop('procs_per_job')
  if len(args) > 4:
    mem = args[4]
  elif 'mem' in kwargs:
    mem = kwargs['mem']
    kwargs.pop('mem')

  if dispatch == mapping['sge']:
    return dispatch(*args[0:2], max_jobs=max_jobs,
        procs_per_job=procs_per_job, **kwargs)
  else:
    return dispatch(*args[0:2], max_jobs=max_jobs,
        procs_per_job=procs_per_job, mem=mem, **kwargs)


def ClusterChecked(function, parameter_list, *args, **kwargs):
  '''Calls ClusterRun and raises an exception if any results return False.'''
  res = ClusterRun(function, parameter_list, *args, **kwargs)
  if all(res):
    print('All', len(res), 'jobs successful.')
  else:
    failed = sum([not bool(b) for b in res])
    if failed == len(res):
      raise RuntimeError('All '+str(failed)+' jobs failed!')
    else:
      print('Error on job parameters:\n  ' + 
          '\n  '.join(str(parameter_list[i]) for i in range(len(res))
            if not bool(res[i])))
      raise RuntimeError(str(failed)+' of '+str(len(res))+' jobs failed!')


def ChunkLabels(count, params=None):
  if params is None:
    chunklen = len(str(count-1))
    return ['chunk{}'.format(str(i).rjust(chunklen, str(0)))
            for i in range(count)]
  else:
    return [label for label,params in ChunkParams(count, params)]


def ChunkParams(count, params):
  Nmin = len(params)//count
  Narr = np.array([Nmin]*min(len(params), count))
  Narr[:len(params)-count*Nmin] += 1
  Nsum = [0] + list(np.cumsum(Narr))
  chunklabels = ChunkLabels(count)
  if hasattr(params, 'iloc'):
    res = [(label, params.iloc[s:s+N]) for label,s,N in
           zip(chunklabels, Nsum, Narr)]
  else:
    res = [(label, params[s:s+N]) for label,s,N in
           zip(chunklabels, Nsum, Narr)]
  return res


class SpectralAnalysis():
  class AutoBuf:
    pass

  def __init__(self, freqs=None, subs=None, dfs=None, electrodes=None,
      elec_masks=None, morlet_reps=6, buf_ms=AutoBuf, bin_Hz=None,
      time_range=(0,1600), event_types=['WORD'], split_recall=True,
      debug=False):


    self.debug = debug
    self.freqs = freqs
    self.bin_Hz = bin_Hz
    self.event_types = event_types
    self.split_recall = split_recall

    self.subjects = None
    if dfs is not None:
      self.by_session = True
      self.df_all = dfs
      self.subjects = self.df_all.subject.unique()
    else:
      self.by_session = False

    if subs is not None:
      if self.by_session:
        raise ValueError('Set subs or dfs, but not both')
      self.subjects = subs

    if self.subjects is None:
      raise ValueError('Set either subs or dfs')


    if electrodes is None and elec_masks is None:
      self.use_all_elecs = True
    else:
      self.use_all_elecs = False

    if electrodes is not None:
      self.electrodes = electrodes
      self.use_elec_masks = False
    else:
      self.use_elec_masks = True

    if elec_masks is not None:
      if not self.use_elec_masks:
        raise ValueError('Set electrodes or elec_masks, but not both')
      self.elec_masks = elec_masks


    self.SetTimeRange(time_range[0], time_range[1])
    self.morlet_reps = morlet_reps

    if buf_ms is self.AutoBuf:
      self.buf_ms_was_auto = True
      if self.freqs is None:
        self.buf_ms = 0
      else:
        self.buf_ms = 1500*(self.morlet_reps/2)/min(self.freqs)
    else:
      self.buf_ms_was_auto = False
      self.buf_ms = buf_ms

    self.avg_ref = False
    self.internal_bipolar = False

    self.sposarr = np.zeros(0, dtype=np.int64)
    self.list_count = 0


  def SetTimeRange(self, left_ms, right_ms):
    self.left_ms = left_ms
    self.right_ms = right_ms
    if self.bin_Hz is not None:
      self.time_elements = int(((right_ms - left_ms)/1000.0)*self.bin_Hz + 0.5)


  def LoadEEG(self, row, mask_index=None):
    '''Sets self.eeg_ptsa as xarray.TimeSeries with: event, channel, time'''
    reader = CMLReadDFRow(row)
    # This does not work for this data set,
    # so we will get these from load_eeg(...).to_ptsa() later.
    #contacts = reader.load('contacts')
    evs = reader.load('events')
    evs_mask = np.zeros(len(evs.type), dtype=bool)
    for et in self.event_types:
      evs_mask |= evs.type == et
    self.enc_evs = evs[evs_mask]
    sposarr = np.array(self.enc_evs.groupby('serialpos')['recalled'].sum())
    if len(sposarr) > len(self.sposarr):
      self.sposarr = self.sposarr.copy()
      self.sposarr.resize(len(sposarr))
    self.sposarr[:len(sposarr)] += sposarr
    self.list_count += len(set(self.enc_evs.list))
    #LogDFErr(row, mask = evs_mask, list_count = self.list_count, enc_evs = self.enc_evs, evs = evs)

    # Use a pairs scheme if it exists.
    self.pairs = None
    # Disabled for now due to montage errors
    try:
      self.pairs = reader.load('pairs')
    except:
      pass

    if 'WORD' in self.event_types:
      if np.sum(self.enc_evs.recalled == True) == 0:
          raise IndexError('No recalled events')
      if np.sum(self.enc_evs.recalled == False) == 0:
          raise IndexError('No non-recalled events')

    if self.pairs is None:
      # clean=True for Localized Component Filtering (LCF)
      eeg = reader.load_eeg(events=self.enc_evs, \
        rel_start=self.left_ms - self.buf_ms, \
        rel_stop=self.right_ms + self.buf_ms, clean=True)
    else:
      # clean=True for Localized Component Filtering (LCF)
      eeg = reader.load_eeg(events=self.enc_evs, \
        rel_start=self.left_ms - self.buf_ms, \
        rel_stop=self.right_ms + self.buf_ms, clean=True)
      if self.pairs.shape[0] != eeg.data.shape[1]:
        eeg = reader.load_eeg(events=self.enc_evs, scheme=self.pairs, \
          rel_start=self.left_ms - self.buf_ms, \
          rel_stop=self.right_ms + self.buf_ms, clean=True)

    if len(eeg.events) != self.enc_evs.shape[0]:
        raise IndexError(str(len(eeg.events)) + \
            ' eeg events for ' + str(self.enc_evs.shape[0]) + \
            ' encoding events')

    if self.avg_ref == True:
        # Reference to average
        avg_ref_data = np.mean(eeg.data, (1))
        for i in range(eeg.data.shape[1]):
            eeg.data[:,i,:] = eeg.data[:,i,:] - avg_ref_data

    if self.internal_bipolar == True:
        # Bipolar reference to nearest labeled electrode
        eeg.data -= np.roll(eeg.data, 1, 1)

    self.sr = eeg.samplerate

    eeg_ptsa = eeg.to_ptsa()

    if self.bin_Hz is None:
      self.trim_start = int((self.buf_ms/1000.0)*self.sr + 0.5)
      self.time_elements = self.eeg_ptsa.shape[2] - 2*self.trim_start
    else:
      self.trim_start = int((self.buf_ms/1000.0)*self.bin_Hz + 0.5)
      # self.time_elements set in __init__
    if self.trim_start < 0:
      self.trim_start = 0
    if self.time_elements < 0:
      self.time_elements = 0

    if self.use_all_elecs:
      self.eeg_ptsa = eeg_ptsa
      self.channel_flags = [True]*len(eeg_ptsa.channel.values)
    else:
      if self.use_elec_masks:
          if self.elec_masks[mask_index] is None:
              raise ValueError( \
                  'No channel mask available for session ' + \
                  str(mask_index))
          if isinstance(self.elec_masks[mask_index], np.ndarray):
              channel_flags = self.elec_masks[mask_index].tolist()
          else:
              channel_flags = self.elec_masks[mask_index]
      else:
          channels = eeg_ptsa.channel.values
          channel_flags = [c in self.electrodes for c in channels]

      if np.sum(channel_flags)==0:
          if self.use_elec_masks:
              raise IndexError('No channels for region index ' + \
                  str(mask_index))
          else:
              raise IndexError('No matching channels found for ' + \
                  str(self.electrodes))

      self.eeg_ptsa = eeg_ptsa[:,channel_flags,:]
      self.channel_flags = channel_flags

    self.channels = self.eeg_ptsa.channel.values


  def FilterLineNoise(self):
    freq_range = [58., 62.]
    b_filter = ButterworthFilter(freq_range=freq_range, filt_type='stop',
        order=4)
    self.eeg_ptsa = b_filter.filter(self.eeg_ptsa)


  def MorletPower(self):
    if self.freqs is None:
      raise ValueError('freqs must be set for Morlet Wavelet analysis')

    wf = morlet.MorletWaveletFilter(freqs=self.freqs,
        width=self.morlet_reps, output=['power'], \
      complete=True)
    # freqs, events, elecs, and time
    self.powers = wf.filter(self.eeg_ptsa)

    if np.any(np.isnan(self.powers)):
      raise ValueError('nan values in Morlet Wavelet power')


  def MorletComplex(self):
    if self.freqs is None:
      raise ValueError('freqs must be set for Morlet Wavelet analysis')

    wf = morlet.MorletWaveletFilter(freqs=self.freqs,
        width=self.morlet_reps, output=['complex'], \
      complete=True)
    # freqs, events, elecs, and time
    self.phasors = wf.filter(self.eeg_ptsa)

    if np.any(np.isnan(self.phasors)):
      raise ValueError('nan values in Morlet Wavelet complex values')


  def ResampleEEG(self):
    '''If set, bin EEG down to bin_Hz as a sampling rate'''
    if (self.bin_Hz == None) or (self.bin_Hz == self.sr):
      return

    self.eeg_ptsa = self.eeg_ptsa.resampled(self.bin_Hz)
    self.sr = self.bin_Hz

  def ResamplePowers(self):
    '''If set, bin powers down to bin_Hz as a sampling rate'''
    if (self.bin_Hz == None) or (self.bin_Hz == self.sr):
      return

    if self.buf_ms < 1000.0/self.sr:
      new_elements = self.time_elements
    else:
      new_elements = int(round(self.powers.shape[3]*(self.bin_Hz/self.sr)))

    self.powers = \
        scipy.signal.resample(self.powers, new_elements, axis=3)
    self.sr = self.bin_Hz

  def ResamplePhasors(self):
    '''If set, bin phasors down to bin_Hz as a sampling rate'''
    if (self.bin_Hz == None) or (self.bin_Hz == self.sr):
      return

    if self.buf_ms < 1000.0/self.sr:
      new_elements = self.time_elements
    else:
      new_elements = int(round(self.phasors.shape[3]*(self.bin_Hz/self.sr)))

    self.phasors = \
        scipy.signal.resample(self.phasors, new_elements, axis=3)
    self.sr = self.bin_Hz


  def TrimEEG(self):
    if self.eeg_ptsa.shape[2] == self.time_elements:
      return

    self.eeg_ptsa = self.eeg_ptsa.remove_buffer(self.buf_ms/1000.0)

    if self.eeg_ptsa.shape[2] != self.time_elements:
      raise RuntimeError('PTSA resampling failed, ' +
          str(self.eeg_ptsa.shape[2]) + '!=' + str(self.time_elements))

  def TrimPowers(self):
    if self.powers.shape[2] == self.time_elements:
      return

    self.powers = self.powers[... ,
        self.trim_start:self.trim_start+self.time_elements]

  def TrimPhasors(self):
    if self.phasors.shape[2] == self.time_elements:
      return

    self.phasors = self.phasors[... ,
        self.trim_start:self.trim_start+self.time_elements]


  def NormalizePhasors(self):
    self.phasors = self.phasors / np.absolute(self.phasors)


  def AddHarmonicPieces1(self, rec, sub_data, phasors):
    sub_data[:,:,:,rec,0] += phasors.shape[1]
    sub_data[:,:,:,rec,1] += np.sum(np.real(phasors), axis=(1))
    sub_data[:,:,:,rec,2] += np.sum(np.imag(phasors), axis=(1))


  def AddHarmonicPieces2(self, rec, sub_data, phasors):
    sub_data[:,:,:,rec,0] += phasors.shape[1]
    sub_data[:,:,:,rec,1] += np.sum(np.real(phasors), axis=(1))
    sub_data[:,:,:,rec,2] += np.sum(np.imag(phasors), axis=(1))
    sub_data[:,:,:,rec,3] += np.sum(np.cos(2*np.angle(phasors)), axis=(1))
    sub_data[:,:,:,rec,4] += np.sum(np.sin(2*np.angle(phasors)), axis=(1))


  def PhaseConsistency(self):
    '''Note:  Requires prevalidation that all data frames for the same
       subject have the same location masks.'''
    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []

    self.Ns = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      elec_cnt = np.sum(self.elec_masks[0])
      # Store count and first and second harmonic sums, then divide.
      # freqs, elec_cnt, time, rec, N/g1/s1/g2/s2
      if self.split_recall:
        sub_data_shape = (len(self.freqs), elec_cnt, self.time_elements, 2, 5)
      else:
        sub_data_shape = (len(self.freqs), elec_cnt, self.time_elements, 1, 5)
      sub_res_data = np.zeros(sub_data_shape)

      lists_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)
          self.FilterLineNoise()
          self.ResampleEEG()
          self.MorletComplex()
          #self.ResamplePhasors()
          self.TrimPhasors()
          self.NormalizePhasors()

          # Gather the harmonics of events/elecs/sess for full circ dispersion.
          if self.split_recall:
            self.AddHarmonicPieces2(0, sub_res_data, \
                self.phasors[:, self.enc_evs.recalled == True, :, :].data)
            self.AddHarmonicPieces2(1, sub_res_data, \
                self.phasors[:, self.enc_evs.recalled == False, :, :].data)
          else:
            self.AddHarmonicPieces2(0, sub_res_data, \
                self.phasors[:, :, :, :].data)

          lists_per_sub += len(set(self.enc_evs.list))

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

      # sub_res_data:
      #   freqs, elecs, time, rec, N/g1/s1/g2/s2

      sub_res_data[:,:,:,:,1:] /= sub_res_data[:,:,:,:,[0,0,0,0]]

      r_sqrd = sub_res_data[:,:,:,:,1]**2 + \
               sub_res_data[:,:,:,:,2]**2
      N = sub_res_data[:,:,:,:,0]
      z = N * r_sqrd
      # 0 low phase consistency, 1 high phase consistency.
      # See PhaseConsistency free function docstring.
      zs_all_elecs = (z-1)/(N-1)
      zs = np.mean(zs_all_elecs, axis=(1))

      if self.split_recall:
        rec_results.append(zs[:,:,0])
        nrec_results.append(zs[:,:,1])
      else:
        results.append(zs[:,:,0])

      self.Ns.append(lists_per_sub)

    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results


  def PhaseConsistencySerPos(self, serialposranges):
    '''Note:  Requires prevalidation that all data frames for the same
       subject have the same location masks.'''
    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []
    Nspos_ranges = len(serialposranges)

    self.Ns = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      elec_cnt = np.sum(self.elec_masks[0])
      # Store count and first and second harmonic sums, then divide.
      # Nspos_ranges, freqs, elecs, time, rec, N/g1/s1/g2/s2
      if self.split_recall:
        sub_data_shape = (Nspos_ranges, len(self.freqs), elec_cnt, \
                          self.time_elements, 2, 5)
      else:
        sub_data_shape = (Nspos_ranges, len(self.freqs), elec_cnt, \
                          self.time_elements, 1, 5)
      sub_res_data = np.zeros(sub_data_shape)

      lists_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)
          self.FilterLineNoise()
          self.ResampleEEG()
          self.MorletComplex()
          #self.ResamplePhasors()
          self.TrimPhasors()
          self.NormalizePhasors()

          # Gather the harmonics of events/elecs/sess for full circ dispersion.
          # Cluster events by serial position ranges
          for sposr in range(Nspos_ranges):
            spos_mask = [s in serialposranges[sposr] for s in \
                         self.enc_evs.serialpos]
            if self.split_recall:
              self.AddHarmonicPieces2(0, sub_res_data[sposr], \
                  self.phasors[:, (self.enc_evs.recalled == True) & \
                  spos_mask, :, :].data)
              self.AddHarmonicPieces2(1, sub_res_data[sposr], \
                  self.phasors[:, (self.enc_evs.recalled == False) & \
                  spos_mask, :, :].data)
            else:
              self.AddHarmonicPieces2(0, sub_res_data[sposr], \
                  self.phasors[:, spos_mask, :, :].data)

          lists_per_sub += len(set(self.enc_evs.list))

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

      # sub_res_data:
      #   Nspos_ranges, freqs, elecs, time, rec, N/g1/s1/g2/s2

      # Convert all the harmonics to means.
      sub_res_data[:,:,:,:,:,1:] /= sub_res_data[:,:,:,:,:,[0,0,0,0]]

      r_sqrd = sub_res_data[:,:,:,:,:,1]**2 + \
               sub_res_data[:,:,:,:,:,2]**2
      N = sub_res_data[:,:,:,:,:,0]
      z = N * r_sqrd
      # 0 low phase consistency, 1 high phase consistency.
      # See PhaseConsistency free function docstring.
      zs_all_elecs = (z-1)/(N-1)
      zs = np.mean(zs_all_elecs, axis=(2))

      if self.split_recall:
        rec_results.append(zs[:,:,:,0])
        nrec_results.append(zs[:,:,:,1])
      else:
        results.append(zs[:,:,:,0])

      self.Ns.append(lists_per_sub)

    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results


  def PhaseSerPos(self, serialposranges):
    '''Note:  Requires prevalidation that all data frames for the same
       subject have the same location masks.'''
    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []
    Nspos_ranges = len(serialposranges)

    self.Ns = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      elec_cnt = np.sum(self.elec_masks[0])
      # Store count and first harmonic sums, then divide.
      # Nspos_ranges, freqs, elecs, time, rec, N/g1/s1
      if self.split_recall:
        sub_data_shape = (Nspos_ranges, len(self.freqs), elec_cnt, \
                          self.time_elements, 2, 3)
      else:
        sub_data_shape = (Nspos_ranges, len(self.freqs), elec_cnt, \
                          self.time_elements, 1, 3)
      sub_res_data = np.zeros(sub_data_shape)

      lists_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)
          self.FilterLineNoise()
          self.ResampleEEG()
          self.MorletComplex()
          #self.ResamplePhasors()
          self.TrimPhasors()
          self.NormalizePhasors()

          # Gather the harmonics of events/elecs/sess for full circ dispersion.
          # Cluster events by serial position ranges
          for sposr in range(Nspos_ranges):
            spos_mask = [s in serialposranges[sposr] for s in \
                         self.enc_evs.serialpos]
            if self.split_recall:
              self.AddHarmonicPieces1(0, sub_res_data[sposr], \
                  self.phasors[:, (self.enc_evs.recalled == True) & \
                  spos_mask, :, :].data)
              self.AddHarmonicPieces1(1, sub_res_data[sposr], \
                  self.phasors[:, (self.enc_evs.recalled == False) & \
                  spos_mask, :, :].data)
            else:
              self.AddHarmonicPieces1(0, sub_res_data[sposr], \
                  self.phasors[:, spos_mask, :, :].data)

          lists_per_sub += len(set(self.enc_evs.list))

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

      # sub_res_data:
      #   Nspos_ranges, freqs, elecs, time, rec, N/g1/s1

      # Convert all the harmonics to means.
      sub_res_data[:,:,:,:,:,1:] /= sub_res_data[:,:,:,:,:,[0,0]]

      phase_data = np.arctan2(
          sub_res_data[:,:,:,:,:,2], sub_res_data[:,:,:,:,:,1])

      if self.split_recall:
        rec_results.append(np.stack(
          [sub_res_data[:,:,:,:,0,0], phase_data[:,:,:,:,0]], axis=4))
        nrec_results.append(np.stack(
          [sub_res_data[:,:,:,:,1,0], phase_data[:,:,:,:,1]], axis=4))
      else:
        results.append(np.stack(
          [sub_res_data[:,:,:,:,0,0], phase_data[:,:,:,:,0]], axis=4))

      self.Ns.append(lists_per_sub)

    # For each results list by subs of arrays of:
    #   Nspos_ranges, freqs, elecs, time, N/phase
    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results


  def NormCent(self, a):
    '''Zero center by mean of last axis, then normalize by stddev.'''
    return ((a.T-a.mean(axis=-1).T) / a.std(axis=-1, ddof=1).T).T


  def AvgNormalizedEEG(self):
    '''Note:  Requires prevalidation that all data frames for the same
       subject have the same location masks.'''
    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []

    self.Ns = []
    restore_buf_ms = self.buf_ms
    if self.buf_ms_was_auto:
      self.buf_ms = 750

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      elec_cnt = np.sum(self.elec_masks[0])
      # Store count and first and second harmonic sums, then divide.
      # rec, elecs, time
      if self.split_recall:
        sub_data_shape = (2, elec_cnt, self.time_elements)
      else:
        sub_data_shape = (1, elec_cnt, self.time_elements)
      sub_res_data = np.zeros(sub_data_shape)

      lists_per_sub = 0
      res_evcnt = 0
      rec_evcnt = 0
      nrec_evcnt = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)
          self.FilterLineNoise()
          self.ResampleEEG()
          self.TrimEEG()

          def NormSumEvents(a):
            normcent = np.sum(self.NormCent(a), axis=0)
            cnt = a.shape[0]
            return normcent, cnt

          # Gather the harmonics of events/elecs/sess for full circ dispersion.
          if self.split_recall:
            # events, elecs, time
            nsum,cnt = NormSumEvents( \
                self.eeg_ptsa[self.enc_evs.recalled == True, :, :])
            sub_res_data[0] += nsum
            rec_evcnt += cnt
            nsum,cnt = NormSumEvents( \
                self.eeg_ptsa[self.enc_evs.recalled == False, :, :])
            sub_res_data[1] += nsum
            nrec_evcnt += cnt
          else:
            nsum,cnt = NormSumEvents( \
                self.eeg_ptsa[:, :, :])
            sub_res_data[0] += nsum
            res_evcnt += cnt

          lists_per_sub += len(set(self.enc_evs.list))

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

      if self.split_recall:
        # Normalize by the number of events
        sub_res_data[0] /= rec_evcnt
        sub_res_data[1] /= nrec_evcnt
        rec_results.append(sub_res_data[0])
        nrec_results.append(sub_res_data[1])
      else:
        sub_res_data[0] /= res_evcnt
        results.append(sub_res_data[0])

      self.Ns.append(lists_per_sub)

    self.buf_ms = restore_buf_ms

    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results


  def PowerByFreqTime(self, zscore=False, bin_elecs=True):
    '''Note:  Requires prevalidation that all data frames for the same
       subject have the same location masks.'''
    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []

    self.Ns = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      elec_cnt = np.sum(self.elec_masks[0])
      # Store count and first and second harmonic sums, then divide.
      # freqs, elec_cnt, time, rec, N/power
      if self.split_recall:
        sub_data_shape = (len(self.freqs), elec_cnt, self.time_elements, 2, 2)
      else:
        sub_data_shape = (len(self.freqs), elec_cnt, self.time_elements, 1, 2)
      sub_res_data = np.zeros(sub_data_shape)

      lists_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)
          self.FilterLineNoise()
          self.ResampleEEG()
          self.MorletPower()
          self.TrimPowers()

          # Z-scoring across events
          if zscore:
            self.powers = scipy.stats.zscore(self.powers, 1, ddof=1)

          # Add up the powers.
          if self.split_recall:
            def AddPow(rec, sub_res_data, select_pows):
              sub_res_data[..., rec, 0] += select_pows.shape[1]
              sub_res_data[..., rec, 1] += np.sum(select_pows, axis=1)

            AddPow(0, sub_res_data,
                self.powers[:, self.enc_evs.recalled == True, ...].data)
            AddPow(1, sub_res_data,
                self.powers[:, self.enc_evs.recalled == False, ...].data)

          else:
            AddPow(0, sub_res_data, self.powers.data)

          lists_per_sub += len(set(self.enc_evs.list))

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

      # sub_res_data:
      #   freqs, elecs, time, rec, N/powers

      if bin_elecs:
        sub_res_data = np.mean(sub_res_data, axis=1)

      # Divide by event count.
      sub_res_data = sub_res_data[..., 1] / sub_res_data[..., 0]
      # sub_res_data now:
      #   freqs, elecs, time, rec


      if self.split_recall:
        rec_results.append(sub_res_data[..., 0])
        nrec_results.append(sub_res_data[..., 1])
      else:
        results.append(sub_res_data[..., 0])

      self.Ns.append(lists_per_sub)

    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results



  def PowerSpectra(self, avg_ref=False, zscore=False, bin_elecs=True, \
        internal_bipolar=False):

    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []

    self.Ns = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      # Store count and first and second harmonic sums, then divide.
      # bin_elecs == False:  rec, freqs
      # bin_elecs == True:  rec, freqs, elecs  (Set later)
      if self.split_recall:
        sub_data_shape = (2, len(self.freqs))
      else:
        sub_data_shape = (1, len(self.freqs))
      sub_res_data = np.zeros(sub_data_shape)

      df_per_sub = 0
      lists_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)

          if avg_ref == True:
            # Reference to average
            avg_ref_data = np.mean(self.eeg_ptsa.data, (1))
            for i in range(self.eeg_ptsa.data.shape[1]):
              self.eeg_ptsa.data[:,i,:] = \
                self.eeg_ptsa.data[:,i,:] - avg_ref_data

          if internal_bipolar == True:
            # Bipolar reference to nearest labeled electrode
            self.eeg_ptsa.data -= np.roll(self.eeg_ptsa.data, 1, 1)

          self.FilterLineNoise()
          self.ResampleEEG()
          self.MorletPower()
          self.TrimPowers()

          # Average over time
          self.powers = np.mean(self.powers, (3))

          # Across events
          if zscore:
            self.powers = scipy.stats.zscore(self.powers, 1, ddof=1)

          if bin_elecs:
            if self.split_recall:
              rec_powers = np.mean(self.powers[:, \
                  self.enc_evs.recalled == True, :].data, (1,2))
              nrec_powers = np.mean(self.powers[:, \
                  self.enc_evs.recalled == False, :].data, (1,2))
            else:
              res_powers = np.mean(self.powers[:, :, :].data, (1,2))
          else:
            if df_per_sub == 0:
              first_channel_flags = self.channel_flags
              num_channels_found = self.powers.shape[2]
              sub_res_data = np.zeros((sub_data_shape[0], \
                sub_data_shape[1], self.powers.shape[2]))
            else:
              if np.any(first_channel_flags != self.channel_flags):
                raise IndexError( \
                  'Mismatched electrodes for subject')
              if num_channels_found != self.powers.shape[2]:
                raise IndexError( \
                  'Inconsistent number of electrodes found')

            if self.split_recall:
              rec_powers = np.mean(self.powers[:, \
                self.enc_evs.recalled == True, :].data, (1))
              nrec_powers = np.mean(self.powers[:, \
                self.enc_evs.recalled == False, :].data, (1))

            else:
              res_powers = np.mean(self.powers[:, :, :].data, (1))

          if np.any(np.isnan(rec_powers)) or \
              np.any(np.isnan(nrec_powers)):
            raise ValueError('nan values in eeg power')

          if self.split_recall:
            sub_res_data[0] += rec_powers
            sub_res_data[1] += nrec_powers
          else:
            sub_res_data[0] += res_powers

          df_per_sub += 1
          lists_per_sub += len(set(self.enc_evs.list))

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

        self.Ns.append(lists_per_sub)

      sub_res_data /= df_per_sub

      if self.split_recall:
        rec_results.append(sub_res_data[0])
        nrec_results.append(sub_res_data[1])
      else:
        results.append(sub_res_data[0])

    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results


  def PowerEventsByFreqsChans(self, avg_ref=False, internal_bipolar=False):

    results = []
    recalls = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      # events, (freqs*chans)
      sub_results = []
      # events
      sub_recalls = []

      df_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)

          if avg_ref == True:
            # Reference to average
            avg_ref_data = np.mean(self.eeg_ptsa.data, (1))
            for i in range(self.eeg_ptsa.data.shape[1]):
              self.eeg_ptsa.data[:,i,:] = \
                self.eeg_ptsa.data[:,i,:] - avg_ref_data

          if internal_bipolar == True:
            # Bipolar reference to nearest labeled electrode
            self.eeg_ptsa.data -= np.roll(self.eeg_ptsa.data, 1, 1)

          self.FilterLineNoise()
          self.ResampleEEG()
          self.MorletPower()
          self.TrimPowers()

          # Average over time
          self.powers = np.mean(self.powers, (3))

          if df_per_sub == 0:
            first_channels = self.channels
          else:
            if np.any(self.channels != first_channels):
              raise IndexError( \
                'Mismatched electrodes for subject')

          # events, freqs, elecs
          swapped_axes = np.swapaxes(self.powers.data, 0, 1)
          feature_shape = (swapped_axes.shape[0], \
            swapped_axes.shape[1]*swapped_axes.shape[2])
          sub_results.append(swapped_axes.reshape(feature_shape))
          swapped_axes = None
          sub_recalls.append(self.enc_evs.recalled)

          df_per_sub += 1

        except Exception as e:
          LogDFException(row, e)
          if self.debug:
            raise

      sub_results = np.array(sub_results)
      # Merge sessions:
      #sub_results = sub_results.reshape((sub_results.shape[0] * \
      #    sub_results.shape[1], sub_results.shape[2]))
      results.append(sub_results)
      recalls.append(np.array(sub_recalls))

    ret_results = np.array(results)
    ret_recalls = np.array(recalls)
    # ret_results: sub, sessions, events, freqs*elecs
    # ret_recalls: sub, sessions, events
    return (ret_results, ret_recalls)

