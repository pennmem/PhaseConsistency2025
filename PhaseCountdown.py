#!/usr/bin/env python3

from CMLTools import *

run = RunOpts(__name__, ['setup', 'params', 'clusterphaseall',
    'clusterconsistency', 'clusterlowrep', 'plots'])

exper_list = ['FR1']
#exper_list = ['catFR1']
#analysis_region = 'hip'
#analysis_region = 'ltc'
analysis_region = 'pfc'
#analysis_region = 'all'
settings_file = f'phase_countdown_{"_".join(exper_list)}_{analysis_region}.json'
if run.setup:
  settings = Settings()
  settings.freqs = np.arange(3, 25, 1)
  settings.exp_list = exper_list
  settings.time_range = (-750, 750)
  settings.bin_Hz = 250
  settings.buf_ms = 1500
  settings.logdir = 'log'
  settings.scheduler = 'slurm'
  total_RAM = 800
  RAM_per_job = {'hip':10, 'ltc':25, 'pfc':36, 'all':80}[analysis_region]
  settings.max_jobs = total_RAM // RAM_per_job
  settings.mem = f'{RAM_per_job}GB'

  settings.reg = analysis_region
  if settings.reg == 'hip':
    settings.regions = Locator(None).hippocampus_regions
    settings.reg_label = 'Hippocampal'
    settings.reg_label2 = 'Hippocampus'
    settings.reg_short = 'Hip.'
  elif settings.reg == 'ltc':
    settings.regions = Locator(None).ltc_regions
    settings.reg_label = 'Lateral Temporal Cortex'
    settings.reg_label2 = 'Lateral Temporal Cortex'
    settings.reg_short = 'LTC'
  elif settings.reg == 'pfc':
    settings.regions = Locator(None).pfc_regions
    settings.reg_label = 'Prefrontal Cortex'
    settings.reg_label2 = 'Prefrontal Cortex'
    settings.reg_short = 'PFC'
  elif settings.reg == 'all':
    settings.regions = None
    settings.reg_label = 'All'
    settings.reg_label2 = 'Whole Brain'
    settings.reg_short = 'WB'
  else:
    raise ValueError(f'Unknown region specified: {settings.reg}')

  settings.exp_str = '_'.join(settings.exp_list)
  settings.expreg = f'{settings.exp_str}_{settings.reg}'

  settings.outdir = f'phase_countdown_{settings.expreg}'
  settings.statsdir = f'stats_countdown_{settings.expreg}'
  settings.logfile = os.path.join(settings.logdir,
      settings.outdir+'.txt')

  settings.Save(settings_file)

  os.makedirs(settings.outdir, exist_ok=True)
  os.makedirs(settings.statsdir, exist_ok=True)
  os.makedirs(settings.logdir, exist_ok=True)

if run.main and not run.setup:
  settings = Settings.Load(settings_file)

if run.plots:
  os.makedirs(f'plots_countdown_{settings.expreg}', exist_ok=True)

if run.main:
  SaveFig(None, f'plots_countdown_{settings.expreg}')

  sub_list_data = JSLoad(f'subject_list_{settings.expreg}.json')
  subs_selected = sub_list_data['subs']
  sub_list_counts = sub_list_data['sub_list_counts']

if run.params:
  print(settings)
  print(len(subs_selected),'subjects', subs_selected)
  print('Starting run', datetime.datetime.now())


def PlotPhaseConsistency(data, title, filenamebase, yrange=None):
    fig = StartFig(figsize=(10,4))
    ax = fig.add_subplot(111)
    if yrange is None:
        cax = ax.matshow(data, aspect='auto',
                      cmap='gnuplot2', origin='lower')
    else:
        cax = ax.matshow(data, aspect='auto',
                      cmap='gnuplot2', vmin=yrange[0], vmax=yrange[1],
                      origin='lower')
    cbar = fig.colorbar(cax)
    cbar.set_label('Phase consistency')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Time (ms)')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('Frequency (Hz)')
    plt.title(title, y=1.02)
    SaveFig(filenamebase)

def BlueBlackRedCMAP():
  import matplotlib.colors
  blues = np.outer(np.linspace(1,0,64), [0,0,1,0]) + \
          np.outer(np.ones(64),[0,0,0,1])
  reds = np.outer(np.linspace(0,1,64)[1:], [1,0,0,0]) + \
         np.outer(np.ones(63),[0,0,0,1])
  cmaparr = np.concatenate([blues, reds])
  return matplotlib.colors.ListedColormap(cmaparr)

def BlueBlackRedWCMAP():
  import matplotlib.colors
  blues = np.outer(np.linspace(1,0,64), [0,0,1,0]) + \
          np.outer(np.ones(64),[0,0,0,1])
  reds = np.outer(np.linspace(0,1,64)[1:], [1,0,0,0]) + \
         np.outer(np.ones(63),[0,0,0,1]) + \
         np.outer(np.linspace(0,1,64)[1:]**4, [0,1,1,0])
  cmaparr = np.concatenate([blues, reds])
  return matplotlib.colors.ListedColormap(cmaparr)


def TTestPlot(res_list, Ns_list, jsfile=None):
  FreqLow = np.argwhere(freqs==3)[0][0]
  FreqHigh = np.argwhere(freqs==24)[0][0]
  pre_exp_arr = []
  post_exp_arr = []
  for res in res_list:
    pre_exposure = np.mean(res[FreqLow:FreqHigh+1,
        0:int((-375-(-750))/4+0.5)+1])
    post_exposure = np.mean(res[FreqLow:FreqHigh+1,
        int((750+125)/4+0.5):int((750+375)/4+0.5)+1])
    pre_exp_arr.append(pre_exposure)
    post_exp_arr.append(post_exposure)

  t, p = WeightedNPairedTTest(post_exp_arr, pre_exp_arr, Ns_list)
  #print('pre', pre_exp_arr)
  #print('post', post_exp_arr)
  print('t = ', t, ', p = ', p, sep='')
  if jsfile is not None:
    JSSave(os.path.join(settings.statsdir, jsfile), {'t':t, 'p':p})
  return t,p


def RunPhaseSpreadAllTogether(sub):
    import numpy as np
    import os
    
    from CMLTools import Settings, SubjectDataFrames, CMLReadDFRow, Locator, \
                         LogDFErr, SetLogger, SpectralAnalysis
    
    try:
        error_suffix = 'alltog_'+sub
        
        LogErr = SetLogger(logfile=settings.logfile, suffix=error_suffix)
                
        df_sub = SubjectDataFrames(sub)
        df_sub = df_sub[[s in settings.exp_list for s in df_sub['experiment']]]
        
        locmasks = []

        sess_cnt=0
        valid_sess = []
        for row in df_sub.itertuples():
            try:
                reader = CMLReadDFRow(row)
                locmask = Locator(reader).Regions(settings.regions)
                locmasks.append(locmask)
                valid_sess.append(sess_cnt)
            except Exception as e:
                LogDFErr(row, e, LogErr=LogErr)
            sess_cnt += 1
        mask_array = np.zeros(len(df_sub), dtype=bool)
        mask_array[valid_sess] = 1
        df_sub = df_sub[mask_array]

        sa = SpectralAnalysis(settings.freqs, dfs=df_sub, elec_masks=locmasks, \
            time_range=settings.time_range, bin_Hz=settings.bin_Hz,
            event_types=['COUNTDOWN', 'COUNTDOWN_START'], split_recall=False, 
            buf_ms=settings.buf_ms)
        results = sa.PhaseConsistency()
        
        JSSave(os.path.join(settings.outdir,
          f'alltog_{settings.expreg}_{sub}.json',),
          {'res':results, 'Ns':sa.Ns})
    except Exception as e:
        LogException(e, error_suffix)
        return False
        
    return True


if run.clusterphaseall:
  ClusterChecked(RunPhaseSpreadAllTogether, subs_selected, settings=settings)


if run.plots:
  freqs = settings.freqs
  data = JSLoad(os.path.join(settings.outdir,
          f'alltog_{settings.expreg}_{subs_selected[0]}.json'))
  res_consistency, Ns = data['res'], data['Ns']
  res_shape = res_consistency[0].shape
  xlabels = np.arange(settings.time_range[0], settings.time_range[1]+1, 125)
  xticks = np.linspace(0, res_shape[1]+1, len(xlabels))
  ylabels = freqs[::3]
  yticks = np.linspace(0, res_shape[0]-1, len(ylabels))

  res_list = []
  Ns_list = []
  res_arr = np.zeros(res_shape)
  lst_cnt = 0
  sub_cnt = 0
  for sub in subs_selected:
      try:
          data = JSLoad(os.path.join(settings.outdir,
                  f'alltog_{settings.expreg}_{sub}.json'))
          res_consistency, Ns = data['res'], data['Ns']
      except:
          print('File missing for '+sub)
          continue
      if np.any(np.isnan(res_consistency)):
          print('Discarding '+sub+' for nan values.')
          continue
      res_arr += res_consistency[0]*Ns[0]
      lst_cnt += Ns[0]
      sub_cnt += 1
      res_list.append(res_consistency[0])
      Ns_list.append(Ns[0])
  print(str(sub_cnt)+' subjects loaded.')

  res_arr /= lst_cnt

  FreqLow = np.argwhere(freqs==3)[0][0]
  FreqHigh = np.argwhere(freqs==24)[0][0]
  res_preword = res_arr[FreqLow:FreqHigh+1, 0:int((-375-(-750))/4+0.5)+1]
  sd = np.std(res_preword, ddof=1)
  premean = np.mean(res_preword)
  print('res_preword mean', premean)
  z_arr = res_arr / sd
  z_mask = np.array(z_arr > 5, dtype=float)
  preword_mask = np.zeros(res_shape, dtype=float)
  preword_mask[FreqLow:FreqHigh+1, 0:int((-375-(-750))/4+0.5)+1] = 1

  res_peak_timecoords = np.argmax(res_arr, axis=1)
#  for i,p in enumerate(res_peak_timecoords):
#    if p < 0.5*res_arr.shape[1] or p > 0.9*res_arr.shape[1]:
#      raise RuntimeError(
#          f'Time peak[{i}]={p} out of expected range.  Examine.')
  peak_dict = {int(k):int(v) for k,v in zip(freqs, res_peak_timecoords)}
  JSSave(os.path.join(settings.statsdir, f'peak_timecoords_{settings.expreg}.json'), peak_dict)


def PlotPhaseConsistency(data, title, filenamebase, yrange=None, contour_masks=None):
    '''contour_masks: {'z > 5': mask, 'preword': preword_mask}'''
    import matplotlib
    fig = StartFig(figsize=(10,4))
    ax = fig.add_subplot(111)
    if yrange is None:
        cax = ax.matshow(data, aspect='auto',
                      cmap='gnuplot2', origin='lower')
    else:
        cax = ax.matshow(data, aspect='auto',
                      cmap='gnuplot2', vmin=yrange[0], vmax=yrange[1],
                      origin='lower')
    cbar = fig.colorbar(cax)
    cbar.set_label('Phase consistency')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Time (ms)')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('Frequency (Hz)')
    handles = []
    if contour_masks is not None:
      colors = ['cyan', 'darkred', 'green']
      for (label, contour), color in zip(contour_masks.items(), colors):
        ax.contour(contour, levels=[0.5], linewidths=[2], colors=color)
        handles.append(matplotlib.patches.Patch(color=color, label=label))
    if len(handles) > 0:
      plt.legend(handles=handles)
    plt.title(title, y=1.02)
    SaveFig(filenamebase)
    
if run.plots:
  PlotPhaseConsistency(res_arr, f'{settings.reg_label} Phase Consistency', f'countdown_alltog_{settings.expreg}_phase_consistency',
      contour_masks={'z > 5':z_mask, 'pre-stimulus':preword_mask})

  TTestPlot(res_list, Ns_list, f'countdown_alltog_ttest_{settings.expreg}.json')


if run.main:
  print('Ran', exper_list, analysis_region)
  print('Runtime', bench.TotalStr())
  print('Synchronizing outputs on NFS.')
  os.system('sync')

