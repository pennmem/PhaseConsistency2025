#!/usr/bin/env python3

from CMLTools import *

run = RunOpts(__name__, ['setup', 'params', 'clusterphaseall',
    'clusterserpos', 'clusterconsistency', 'clusterlowrep', 'plots'])

exper_list = ['FR1']
#exper_list = ['catFR1']
#analysis_region = 'hip'
#analysis_region = 'ltc'
#analysis_region = 'pfc'
analysis_region = 'all'
settings_file = f'phase_sessions_{"_".join(exper_list)}_{analysis_region}.json'
if run.setup:
  settings = Settings()
  settings.freqs = np.arange(3, 25, 1)
  settings.exp_list = exper_list
  settings.time_range = (-750, 750)
  settings.bin_Hz = 250
  settings.spos_ranges = [[1,2,3],[4,5,6,7,8,9],[10,11,12]]
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

  settings.outdir = f'phasesessions_{settings.expreg}'
  settings.statsdir = f'stats_{settings.expreg}'
  settings.logfile = os.path.join(settings.logdir,
      settings.outdir+'.txt')

  settings.Save(settings_file)

  os.makedirs(settings.outdir, exist_ok=True)
  os.makedirs(settings.statsdir, exist_ok=True)
  os.makedirs(settings.logdir, exist_ok=True)

if run.main and not run.setup:
  settings = Settings.Load(settings_file)

if run.plots:
  os.makedirs(f'plots_{settings.expreg}', exist_ok=True)

if run.main:
  SaveFig(None, f'plots_{settings.expreg}')

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
            time_range=settings.time_range, bin_Hz=settings.bin_Hz, split_recall=False, 
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
  for i,p in enumerate(res_peak_timecoords):
    if p < 0.5*res_arr.shape[1] or p > 0.9*res_arr.shape[1]:
      raise RuntimeError(
          f'Time peak[{i}]={p} out of expected range.  Examine.')
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
  PlotPhaseConsistency(res_arr, f'{settings.reg_label} Phase Consistency', f'alltog_{settings.expreg}_phase_consistency',
      contour_masks={'z > 5':z_mask, 'preword':preword_mask})

  TTestPlot(res_list, Ns_list, f'alltog_ttest_{settings.expreg}.json')


def RunPhaseConsistencySerPos(sub):
    import numpy as np
    import os
    
    from CMLTools import Settings, SubjectDataFrames, CMLReadDFRow, Locator, \
                         LogDFErr, SetLogger, SpectralAnalysis
    
    try:
        error_suffix = 'serpos_'+sub
        
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
            time_range=(-750, 750), bin_Hz=250, buf_ms=settings.buf_ms)
        results = sa.PhaseConsistencySerPos(settings.spos_ranges)

        JSSave(os.path.join(settings.outdir,
            f'consistency_serpos_{settings.expreg}_{sub}.json'),
            {'res':results, 'Ns':sa.Ns})
        JSSave(os.path.join(settings.outdir,
            f'spos_sess_recall_{settings.expreg}_{sub}.json'),
            {'sposarr':sa.sposarr, 'list_count':sa.list_count})
    except Exception as e:
        LogErr(e)
        return False
        
    return True


if run.clusterserpos:
  ClusterChecked(RunPhaseConsistencySerPos, subs_selected, settings=settings)


if run.plots:
  freqs = settings.freqs
  spos_ranges = settings.spos_ranges
  data = JSLoad(os.path.join(settings.outdir,
      f'consistency_serpos_{settings.expreg}_{subs_selected[0]}.json'))
  res_consistency = data['res']
  Ns = data['Ns']
  spos_len = len(spos_ranges)
  spos_labels = [str(s[0])+'-'+str(s[-1]) if len(s)>1 else str(s[0])
      for s in spos_ranges]
  rec_consistency = res_consistency[0]
  nrec_consistency = res_consistency[1]
  rec_shape = rec_consistency[0][0].shape
  nrec_shape = nrec_consistency[0][0].shape
  xlabels = np.arange(settings.time_range[0], settings.time_range[1]+1, 125)
  xticks = np.linspace(0, rec_consistency[0][0].shape[1]+1, len(xlabels))
  ylabels = freqs[::3]
  yticks = np.linspace(0, rec_consistency[0][0].shape[0]-1, len(ylabels))
  peak_dict = JSLoad(os.path.join(settings.statsdir, f'peak_timecoords_{settings.expreg}.json'))
  res_peak_timecoords = list(peak_dict.values())
  for i,p in enumerate(res_peak_timecoords):
    if p < 0.5*rec_shape[1] or p > 0.9*rec_shape[1]:
      raise RuntimeError(
          f'Loaded time peak[{i}]={p} out of expected range.  Examine.')

  t_dict = {}
  p_dict = {}
  p_arr = []
  diff_spos = []
  diff_spos_freq = []
  for spos in range(spos_len):
      sub_recs = {}
      sub_nrecs = {}
      sub_diffs = {}
      sub_lst_cnt = {}
      rec_arr = np.zeros(rec_shape)
      nrec_arr = np.zeros(nrec_shape)
      lst_cnt = 0
      sub_cnt = 0
      diff_list = []
      diff_subs = []
      diff_subs_freq = []
      Ns_list = []
      for sub in subs_selected:
          try:
              data = JSLoad(os.path.join(settings.outdir,
                  f'consistency_serpos_{settings.expreg}_{sub}.json'))
              rec_consistency = data['res'][0]
              nrec_consistency = data['res'][1]
              Ns = data['Ns']
          except:
              print('File missing for '+sub)
              continue

          if np.any(np.isnan(rec_consistency)) or \
             np.any(np.isnan(nrec_consistency)):

              print('Discarding '+sub+' for nan values.')
              continue

          rec_v = rec_consistency[0][spos]
          nrec_v = nrec_consistency[0][spos]
          rec_arr += rec_v*Ns[0]
          nrec_arr += nrec_v*Ns[0]
          a = sub_recs.setdefault(sub, np.zeros(rec_shape))
          a += rec_v
          a = sub_nrecs.setdefault(sub, np.zeros(nrec_shape))
          a += nrec_v
          diff_v = rec_v - nrec_v
          diff_list.append(diff_v)
          a = sub_diffs.setdefault(sub, np.zeros(rec_shape))
          a += diff_v
          Ns_list.append(Ns[0])
          lst_cnt += Ns[0]
          sub_lst_cnt[sub] = sub_lst_cnt.get(sub, 0) + Ns[0]
          sub_cnt += 1

          FreqLow = np.argwhere(freqs==3)[0][0]
          FreqHigh = np.argwhere(freqs==24)[0][0]
          rec_post_exposure = np.mean(rec_v[FreqLow:FreqHigh+1,
              int((750+125)/4+0.5):int((750+500)/4+0.5)+1])
          nrec_post_exposure = np.mean(nrec_v[FreqLow:FreqHigh+1,
              int((750+125)/4+0.5):int((750+500)/4+0.5)+1])
          diff_subs.append(rec_post_exposure - nrec_post_exposure)

          first_freq = 3
          last_freq = 24
          fstart = settings.freqs.tolist().index(first_freq)
          fend = settings.freqs.tolist().index(last_freq)

          # Check the time bands where phase consistency actually happens for
          # unbiased frequency analysis of changes in phase consistency.
          def FreqRangeTimeAvg(arr):
            selected = []
            for f in range(fstart, fend+1):
              freq = settings.freqs[f]
              f_tstart = int(375*(125-375/freq + 750)/1500)
              f_tend = int(375*(125+375/freq+1000/freq + 750)/1500)
              selected.append(np.mean(arr[f, f_tstart:f_tend+1]))
            return np.array(selected)

          rec_post_freq = FreqRangeTimeAvg(rec_v)
          nrec_post_freq = FreqRangeTimeAvg(nrec_v)
          diff_subs_freq.append(rec_post_freq - nrec_post_freq)

      print(str(sub_cnt)+' subjects loaded.')

      rec_arr /= lst_cnt
      nrec_arr /= lst_cnt
      t,p = TTestPlot(diff_list, Ns_list)
      t_dict.setdefault('_'.join(str(e) for e in spos_ranges[spos]), []).append(t)
      p_dict.setdefault('_'.join(str(e) for e in spos_ranges[spos]), []).append(p)
      p_arr.append(p)

      diff_spos.append(diff_subs)
      diff_spos_freq.append(diff_subs_freq)

      PlotPhaseConsistency(rec_arr,
          f'{settings.reg_short}, Recalled, Ser. Pos. {spos_labels[spos]}',
          f'spos_recalled_phase_consistency_{settings.expreg}_{spos_labels[spos]}')

      PlotPhaseConsistency(nrec_arr,
          f'{settings.reg_short}, Not Recalled, Ser. Pos. {spos_labels[spos]}',
          f'spos_notrecalled_phase_consistency_{settings.expreg}_{spos_labels[spos]}')

      PlotPhaseConsistency(rec_arr-nrec_arr,
          f'{settings.reg_short}, Recalled - Not Recalled, Ser. Pos. {spos_labels[spos]}', \
          f'delta_spos_phase_consistency_{settings.expreg}_{spos_labels[spos]}', yrange=[0, 0.0035])

      # Each subject weighted equally.

      srec_arr = np.array([e for e in sub_recs.values()])
      snrec_arr = np.array([e for e in sub_nrecs.values()])
      sdelta_arr = srec_arr - snrec_arr
      #sdelta_masked = np.zeros(sdelta_arr.shape)
      #sdelta_masked[:,:,sdelta_masked.shape[-1]//2:] = \
      #  sdelta_arr[:,:,sdelta_arr.shape[-1]//2:]
      loaded_subs = list(sub_recs.keys())
      #sdelta_filtered = sdelta_masked[np.array(Ns_list)>=100]
      #delta_peak_freqs = [np.argmax(s)//s.shape[-1] for s in sdelta_masked]
      #delta_peak_freqs = [np.argmax(np.mean(s, axis=1)) for s in sdelta_masked]
      #delta_peak_freqs = [np.argmax(np.mean(s, axis=1)) for s in sdelta_filtered]

      # For all frequencies for this one.
      preword_mask = np.zeros(sdelta_arr.shape[2:], dtype=bool)
      preword_mask[0:int((-375-(-750))/4+0.5)+1] = True
      postword_mask = np.zeros(sdelta_arr.shape[2:], dtype=bool)
      postword_mask[int((125-(-750))/4+0.5):] = True

      # [subs, freq, time]
      sdelta_preword = sdelta_arr[:, :, preword_mask]
      #print('sdelta_preword.shape', sdelta_preword.shape)
      sdelta_postword = sdelta_arr[:, :, postword_mask]
      # [subs, freq]
      sdelta_pre_subfreq = np.mean(sdelta_preword, axis=2)
      sdelta_post_subfreq = np.mean(sdelta_postword, axis=2)
      #print('sdelta_pre_subfreq.shape', sdelta_pre_subfreq.shape)
      sdelta_pre_meanbyfreq = np.mean(sdelta_pre_subfreq, axis=0)
      sdelta_post_meanbyfreq = np.mean(sdelta_post_subfreq, axis=0)

      pltmin = -0.010
      pltmax = 0.010
      StartFig()
      ax1 = plt.gca()
      interval_heights = ConfidenceIntervals(sdelta_post_subfreq)
      interval_min = [e-h for e,h in zip(sdelta_post_meanbyfreq, interval_heights)]
      interval_max = [e+h for e,h in zip(sdelta_post_meanbyfreq, interval_heights)]
      ax1.fill_between(freqs, interval_min, interval_max, color='blue',
                alpha=0.4)
  #    plt.errorbar(freqs, sdelta_peaks_mean,
  #        ConfidenceIntervals(sdelta_peaks), fmt='.')
      ax1.plot(freqs, sdelta_post_meanbyfreq)
      ax1.set_ylabel('Subject Phase Consistency')
      plt.xlabel('frequency')
      plt.ylim((pltmin, pltmax))
      plt.hlines([0], min(freqs), max(freqs), linestyles='dotted')
      plt.title(f'Recalled - Not Recalled, Post-word means, {spos_labels[spos]}')
      plt.tight_layout()
      SaveFig(f'sdelta_means_{settings.expreg}_{spos_labels[spos]}')

      # Peak version - Peaks address biased time sampling on avg
      # Each subject weighted equally.

      pltmin = -0.010
      pltmax = 0.010
      print('sdelta_arr.shape', sdelta_arr.shape)
      sdelta_peaks = \
          np.array([sdelta_arr[:,i,e] for i,e in enumerate(res_peak_timecoords)]).T

      print('10Hz', res_peak_timecoords[7], sdelta_peaks[7])
      sdelta_peaks_mean = np.mean(sdelta_peaks, axis=0)
      StartFig()
      ax1 = plt.gca()
      interval_heights = ConfidenceIntervals(sdelta_peaks)
      interval_min = [e-h for e,h in zip(sdelta_peaks_mean, interval_heights)]
      interval_max = [e+h for e,h in zip(sdelta_peaks_mean, interval_heights)]
      ax1.fill_between(freqs, interval_min, interval_max, color='blue',
                alpha=0.4)
  #    plt.errorbar(freqs, sdelta_peaks_mean,
  #        ConfidenceIntervals(sdelta_peaks), fmt='.')
      ax1.plot(freqs, sdelta_peaks_mean)
      ax1.set_ylabel('Subject Phase Consistency')
      plt.xlabel('frequency')
      plt.ylim((pltmin, pltmax))
      plt.hlines([0], min(freqs), max(freqs), linestyles='dotted')
      plt.title(f'Recalled - Not Recalled, at Standard Peak Times, {spos_labels[spos]}')
      plt.tight_layout()
      SaveFig(f'sdelta_peaks_{settings.expreg}_{spos_labels[spos]}')


      for i in range(0, 8):
        #print(sdelta_arr[i,0,:], loaded_subs[i])
        PlotPhaseConsistency(sdelta_arr[i],
            f'{settings.reg_label2}, Recalled - Not Recalled, Ser. Pos. {spos_labels[spos]}, {loaded_subs[i]}',
            f'delta_spos_{settings.expreg}_{loaded_subs[i]}_{spos_labels[spos]}')

  # serpos, sub
  diff_spos = np.array(diff_spos)
  StartFig(figsize=(5,4))
  interval_heights = [ConfidenceIntervals(d_subs) for d_subs in diff_spos]
  diff_means = np.mean(diff_spos, axis=1)
  print('diff_means', diff_means.shape, diff_means)
  spos_labels = [f'{r[0]}-{r[-1]}' for r in settings.spos_ranges]

  # Add merged
  diff_merged = np.mean(diff_spos, axis=0)
  diff_means = np.array([*diff_means, np.mean(diff_merged)])
  interval_heights.append(ConfidenceIntervals(diff_merged))
  spos_labels.append('merged')

  plt.errorbar(spos_labels, diff_means, interval_heights, fmt='o',
      capsize=5)
  plt.axhline(y=0, color='grey', linestyle='dotted')
  plt.xlabel('Serial Position Group')
  plt.ylabel('Phase Consistency, Rec. - Not Rec.')
  plt.tight_layout()
  SaveFig(f'sme_by_spos_{settings.expreg}')

  # serpos, sub, freq
  diff_spos_freq = np.array(diff_spos_freq)
  # freq, sub
  diff_freq_sub = np.mean(diff_spos_freq, axis=0).T
  print('diff_spos_freq.shape', diff_spos_freq.shape)
  print('diff_freq_sub.shape', diff_freq_sub.shape)
  # freq
  interval_heights = [ConfidenceIntervals(d_subs) for d_subs in diff_freq_sub]
  print('Ns_list', Ns_list)
  print('lst_cnt', lst_cnt)
  # freq
  diff_means = np.sum(diff_freq_sub * Ns_list / lst_cnt, axis=1)
  t, p = WeightedNPairedTTest(diff_freq_sub.T, np.zeros(diff_freq_sub.T.shape), Ns_list)
  print('Weighted t-test: t =', t, 'p =', p)
  # Benjamini-Hochberg FDR correction
  res = statsmodels.stats.multitest.multipletests(p, 0.05, method='fdr_bh')
  FDR_sig = res[0]
  sig_ind = np.argwhere(FDR_sig).ravel().tolist()
  StartFig()
  interval_min = [e-h for e,h in zip(diff_means, interval_heights)]
  interval_max = [e+h for e,h in zip(diff_means, interval_heights)]
  plt.fill_between(settings.freqs, interval_min, interval_max, color='blue',
                alpha=0.4)
  plt.plot(settings.freqs, diff_means, color='blue',
      label='Post-pre phase consistency difference')
  plt.plot(settings.freqs, diff_means, ' ', markevery=sig_ind, marker='*',
      markersize=10, markeredgecolor='m', markerfacecolor='m',
      label='FDR Sig. weighted paired t-test')
  plt.hlines([0], min(freqs), max(freqs), linestyles='dotted')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Phase consistency difference')
  plt.legend()
  plt.tight_layout()
  SaveFig(f'sme_by_freq_{settings.expreg}')


  JSSave(os.path.join(settings.statsdir, f't_p_delta_spos_{settings.expreg}.json'), {'t':t_dict, 'p':p_dict})

  for spos in range(spos_len):
    rec_list = []
    nrec_list = []
    Ns_list = []
    diff_list = []
    rec_arr = np.zeros(rec_shape)
    nrec_arr = np.zeros(nrec_shape)
    lst_cnt = 0
    for sub in subs_selected:
        try:
            data = JSLoad(os.path.join(settings.outdir,
                f'consistency_serpos_{settings.expreg}_{sub}.json'))
            rec_consistency = data['res'][0]
            nrec_consistency = data['res'][1]
            Ns = data['Ns']
        except:
            print('File missing for '+sub)
            continue
        if np.any(np.isnan(rec_consistency)) or np.any(np.isnan(nrec_consistency)):
            print('Discarding '+sub+' for nan values.')
            continue
        rec_arr += rec_consistency[0][spos]*Ns[0]
        nrec_arr += nrec_consistency[0][spos]*Ns[0]
        lst_cnt += Ns[0]
        nrec_list.append(nrec_consistency[0][spos])
        rec_list.append(rec_consistency[0][spos])
        Ns_list.append(Ns[0])
        diff_list.append(rec_consistency[0][spos]-nrec_consistency[0][spos])
    print(str(sub_cnt)+' subjects loaded.')

    rec_arr /= lst_cnt
    nrec_arr /= lst_cnt
    diff_arr = rec_arr-nrec_arr

    PlotPhaseConsistency(rec_arr, f'{settings.reg_short}, Recalled, {settings.reg_label2}, {spos_labels[spos]}',
                f'consistency_recalled_phase_consistency_{settings.expreg}_{spos_labels[spos]}')

    PlotPhaseConsistency(nrec_arr, f'{settings.reg_short}, Not Recalled, {settings.reg_label2}, {spos_labels[spos]}',
                f'consistency_notrecalled_phase_consistency_{settings.expreg}_{spos_labels[spos]}')

    PlotPhaseConsistency(diff_arr, f'{settings.reg_short}, Recalled - Not Recalled, {settings.reg_label2}, {spos_labels[spos]}',
                f'delta_consistency_phase_consistency_{settings.expreg}_{spos_labels[spos]}', yrange=[0, 0.003])

    TTestPlot(diff_list, Ns_list, f'delta_consistency_ttest_{settings.expreg}_{spos_labels[spos]}.json')


    first_freq = 3
    last_freq = 24
    fstart = settings.freqs.tolist().index(first_freq)
    fend = settings.freqs.tolist().index(last_freq)
    tstart = int(375*(125+750)/1500)
    tend = int(375*(500+750)/1500)
    print('tstart/tend:', tstart, tend, diff_arr.shape)


    # Check the time bands where phase consistency actually happens for
    # changes in phase consistency.
    def FreqRangeTimeAvg(arr):
      selected = []
      for f in range(fstart, fend+1):
        freq = settings.freqs[f]
        f_tstart = int(375*(125-375/freq + 750)/1500)
        f_tend = int(375*(125+375/freq+1000/freq + 750)/1500)
        selected.append(np.mean(arr[f, f_tstart:f_tend+1]))
      return np.array(selected)
      #return np.mean(arr[fstart:fend+1, tstart:tend+1], axis=(1))


    rec_freq = FreqRangeTimeAvg(rec_arr)
    nrec_freq = FreqRangeTimeAvg(nrec_arr)
    diff_freq = FreqRangeTimeAvg(diff_arr)

    rec_freq_list = []
    nrec_freq_list = []
    diff_freq_list = []
    for rec,nrec in zip(rec_list, nrec_list):
      avg_rec = FreqRangeTimeAvg(rec)
      avg_nrec = FreqRangeTimeAvg(nrec)
      rec_freq_list.append(avg_rec)
      nrec_freq_list.append(avg_nrec)
      diff_freq_list.append(avg_rec - avg_nrec)

    weights = np.array(Ns_list)/np.mean(Ns_list)
    t, p = WeightedNPairedTTest(rec_freq_list, nrec_freq_list, Ns_list)
    print('Weighted t-test: t =', t, 'p =', p)
    # Benjamini-Hochberg FDR correction
    res = statsmodels.stats.multitest.multipletests(p, 0.05, method='fdr_bh')
    FDR_sig = res[0]
    pval_corr = res[1]
    print('False discovery rate corrected:', FDR_sig)
    print('Corrected pvals:', pval_corr)
    print('Number significant:', np.sum(FDR_sig))

    desc_rec = statsmodels.stats.weightstats.DescrStatsW(rec_freq_list, weights, ddof=1)
    desc_nrec = statsmodels.stats.weightstats.DescrStatsW(nrec_freq_list, weights, ddof=1)
    desc_diff = statsmodels.stats.weightstats.DescrStatsW(diff_freq_list, weights, ddof=1)
    rec_conf_low, rec_conf_high = desc_rec.tconfint_mean()
    nrec_conf_low, nrec_conf_high = desc_nrec.tconfint_mean()
    diff_conf_low, diff_conf_high = desc_diff.tconfint_mean()

    print(rec_freq.shape)


    xax = np.arange(first_freq, last_freq+1)

    # Note to future me et al: These confidence intervals and significance
    # markers appear suspiciously mismatched, but aren't.  This arises because
    # it is a confidence interval for the unpaired recall/not-recall, whereas
    # the t-test is paired, and the recall and not recall covary.
    StartFig()
    plt.plot(xax, rec_freq, color='blue', label='Recalled')
    plt.fill_between(xax, rec_conf_low, rec_conf_high, color='blue', alpha=0.4)
    plt.plot(xax, nrec_freq, color='red', label='Not Recalled')
    plt.fill_between(xax, nrec_conf_low, nrec_conf_high, color='red', alpha=0.4)
    sig_ind = np.argwhere(FDR_sig).ravel().tolist()
    plt.plot(xax, nrec_freq, ' ', markevery=sig_ind, marker='*', \
             markersize=10, markeredgecolor='m', markerfacecolor='m')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase consistency')
    plt.title(f'{settings.reg_label} phase consistency, {spos_labels[spos]}')
    plt.legend()
    SaveFig(f'consistency_recall_notrecall_{settings.expreg}_{spos_labels[spos]}')

    pltmin = -0.0018 if settings.reg=='all' else -0.0098
    pltmax = 0.0045 if settings.reg=='all' else 0.0125
    StartFig()
    plt.axhline(y=0, color='grey', linestyle='dotted')
    plt.plot(xax, diff_freq, color='blue', label='Recalled - Not-recalled')
    plt.fill_between(xax, diff_conf_low, diff_conf_high, color='blue', alpha=0.4)
    sig_ind = np.argwhere(FDR_sig).ravel().tolist()
    plt.plot(xax, diff_freq, ' ', markevery=sig_ind, marker='*', \
             markersize=10, markeredgecolor='m', markerfacecolor='red',
             label='FDR Sig. weighted paired t-test')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase consistency difference')
    plt.title(f'{settings.reg_short} Phase Consistency SME, {spos_labels[spos]}')
    plt.ylim((pltmin, pltmax))
    plt.legend()
    plt.tight_layout()
    SaveFig(f'consistency_diff_{settings.expreg}_{spos_labels[spos]}')

if run.plots:
  res = statsmodels.stats.multitest.multipletests(p_arr, 0.05, method='fdr_bh')
  FDR_sig = res[0]
  pval_corr = res[1]
  print('False discovery rate corrected:', FDR_sig)
  print('Corrected pvals:', pval_corr)
  print('Number significant:', np.sum(FDR_sig))


def RunPhaseConsistency(sub):
    import numpy as np
    import os
    
    from CMLTools import Settings, SubjectDataFrames, CMLReadDFRow, Locator, \
                         LogDFErr, SetLogger, SpectralAnalysis
    
    try:
        error_suffix = 'consistency_'+sub
        
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
            buf_ms=settings.buf_ms)
        (rec_results, nrec_results) = sa.PhaseConsistency()

        JSSave(os.path.join(settings.outdir,
            f'consistency_{settings.expreg}_{sub}.json'),
            {'rec':rec_results, 'nrec':nrec_results, 'Ns':sa.Ns})
    except Exception as e:
        LogErr(e)
        return False
        
    return True


if run.clusterconsistency:
  ClusterChecked(RunPhaseConsistency, subs_selected, settings=settings)


if run.plots:
  freqs = settings.freqs
  data = JSLoad(os.path.join(settings.outdir,
          f'consistency_{settings.expreg}_{subs_selected[0]}.json'))
  rec_consistency, nrec_consistency, Ns = data['rec'], data['nrec'], data['Ns']
  rec_shape = rec_consistency[0].shape
  nrec_shape = nrec_consistency[0].shape
  xlabels = np.arange(settings.time_range[0], settings.time_range[1]+1, 125)
  xticks = np.linspace(0, rec_shape[1]+1, len(xlabels))
  ylabels = freqs[::3]
  yticks = np.linspace(0, rec_shape[0]-1, len(ylabels))

  rec_list = []
  nrec_list = []
  Ns_list = []
  diff_list = []
  rec_arr = np.zeros(rec_shape)
  nrec_arr = np.zeros(nrec_shape)
  lst_cnt = 0
  for sub in subs_selected:
      try:
          data = JSLoad(os.path.join(settings.outdir,
              f'consistency_{settings.expreg}_{sub}.json'))
          rec_consistency, nrec_consistency, Ns = data['rec'], data['nrec'], data['Ns']
      except:
          print('File missing for '+sub)
          continue
      if np.any(np.isnan(rec_consistency)) or np.any(np.isnan(nrec_consistency)):
          print('Discarding '+sub+' for nan values.')
          continue
      rec_arr += rec_consistency[0]*Ns[0]
      nrec_arr += nrec_consistency[0]*Ns[0]
      lst_cnt += Ns[0]
      nrec_list.append(nrec_consistency[0])
      rec_list.append(rec_consistency[0])
      Ns_list.append(Ns[0])
      diff_list.append(rec_consistency[0]-nrec_consistency[0])
  print(str(sub_cnt)+' subjects loaded.')

  rec_arr /= lst_cnt
  nrec_arr /= lst_cnt
  diff_arr = rec_arr-nrec_arr

  PlotPhaseConsistency(rec_arr, f'{settings.reg_short}, Recalled, {settings.reg_label2}',
              f'consistency_recalled_phase_consistency_{settings.expreg}')

  PlotPhaseConsistency(nrec_arr, f'{settings.reg_short}, Not Recalled, {settings.reg_label2}',
              f'consistency_notrecalled_phase_consistency_{settings.expreg}')

  PlotPhaseConsistency(diff_arr, f'{settings.reg_short}, Recalled - Not Recalled, {settings.reg_label2}',
              f'delta_consistency_phase_consistency_{settings.expreg}',
              yrange=[0, 0.003])

  TTestPlot(diff_list, Ns_list, f'delta_consistency_ttest_{settings.expreg}.json')


def PlotComparison(freqi):
  fig = StartFig()
  ax = fig.add_subplot(111)
  ax.plot(rec_arr[freqi], label='Recalled')
  ax.plot(nrec_arr[freqi], label="Not recalled")
  ax.set_xticks(xticks)
  ax.set_xticklabels(xlabels)
  ax.set_xlabel('Time (ms)')
  ax.xaxis.set_ticks_position('bottom')
  ax.set_ylabel('Phase consistency')
  ax.set_title(f'{settings.reg_short}, {settings.freqs[freqi]} Hz')
  ax.legend()
  SaveFig(f'test_fig_{settings.expreg}_{freqi}')


if run.plots:
  PlotComparison(1)
  PlotComparison(4)
  PlotComparison(8)
  first_freq = 3
  last_freq = 24
  fstart = settings.freqs.tolist().index(first_freq)
  fend = settings.freqs.tolist().index(last_freq)
  #tstart = int(375*(0+750)/1500)
  #tend = int(375*(500+750)/1500)
  #def FreqRangeTimeAvg(arr):
  #  return np.mean(arr[fstart:fend+1, tstart:tend+1], axis=(1))

  tstart = int(375*(125+750)/1500)
  tend = int(375*(500+750)/1500)
  print('tstart/tend:', tstart, tend, diff_arr.shape)

  # Check the time bands where phase consistency actually happens for
  # changes in phase consistency.
  def FreqRangeTimeAvg(arr):
    selected = []
    for f in range(fstart, fend+1):
      freq = settings.freqs[f]
      f_tstart = int(375*(125-375/freq + 750)/1500)
      f_tend = int(375*(125+375/freq+1000/freq + 750)/1500)
      selected.append(np.mean(arr[f, f_tstart:f_tend+1]))
    return np.array(selected)


if run.plots:
  rec_freq = FreqRangeTimeAvg(rec_arr)
  nrec_freq = FreqRangeTimeAvg(nrec_arr)
  diff_freq = FreqRangeTimeAvg(diff_arr)

  rec_freq_list = []
  nrec_freq_list = []
  diff_freq_list = []
  for rec,nrec in zip(rec_list, nrec_list):
    avg_rec = FreqRangeTimeAvg(rec)
    avg_nrec = FreqRangeTimeAvg(nrec)
    rec_freq_list.append(avg_rec)
    nrec_freq_list.append(avg_nrec)
    diff_freq_list.append(avg_rec - avg_nrec)

  weights = np.array(Ns_list)/np.mean(Ns_list)
  t, p = WeightedNPairedTTest(rec_freq_list, nrec_freq_list, Ns_list)
  print('Weighted t-test: t =', t, 'p =', p)
  # Benjamini-Hochberg FDR correction
  res = statsmodels.stats.multitest.multipletests(p, 0.05, method='fdr_bh')
  FDR_sig = res[0]
  pval_corr = res[1]
  print('False discovery rate corrected:', FDR_sig)
  print('Corrected pvals:', pval_corr)
  print('Number significant:', np.sum(FDR_sig))

  desc_rec = statsmodels.stats.weightstats.DescrStatsW(rec_freq_list, weights, ddof=1)
  desc_nrec = statsmodels.stats.weightstats.DescrStatsW(nrec_freq_list, weights, ddof=1)
  desc_diff = statsmodels.stats.weightstats.DescrStatsW(diff_freq_list, weights, ddof=1)
  rec_conf_low, rec_conf_high = desc_rec.tconfint_mean()
  nrec_conf_low, nrec_conf_high = desc_nrec.tconfint_mean()
  diff_conf_low, diff_conf_high = desc_diff.tconfint_mean()

  print(rec_freq.shape)


def PlotConsistencyFreq(arr, title):
    xax = np.arange(first_freq, last_freq+1)
    plt.figure()
    plt.plot(xax, arr)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase consistency (radians)')
    plt.title(title)


if run.plots:
  #PlotConsistencyFreq(rec_freq, 'Recalled')
  #PlotConsistencyFreq(nrec_freq, 'Not Recalled')

  xax = np.arange(first_freq, last_freq+1)
  #plt.figure()
  #plt.plot(xax, rec_freq, color='red', label='Recalled')
  #plt.plot(xax, nrec_freq, color='blue', label='Not Recalled')
  #plt.xlabel('Frequency (Hz)')
  #plt.ylabel('Phase consistency (radians)')
  #plt.legend()
  #plt.title('Phase consistency across frequencies, 125ms to 375ms')
  #plt.savefig('consistency_recall_notrecall.png')

  # Note to future me et al: These confidence intervals and significance
  # markers appear suspiciously mismatched, but aren't.  This arises because
  # it is a confidence interval for the unpaired recall/not-recall, whereas
  # the t-test is paired, and the recall and not recall covary.
  StartFig()
  plt.plot(xax, rec_freq, color='blue', label='Recalled')
  plt.fill_between(xax, rec_conf_low, rec_conf_high, color='blue', alpha=0.4)
  plt.plot(xax, nrec_freq, color='red', label='Not Recalled')
  plt.fill_between(xax, nrec_conf_low, nrec_conf_high, color='red', alpha=0.4)
  sig_ind = np.argwhere(FDR_sig).ravel().tolist()
  plt.plot(xax, nrec_freq, ' ', markevery=sig_ind, marker='*', \
           markersize=10, markeredgecolor='m', markerfacecolor='m')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Phase consistency')
  plt.title(f'{settings.reg_label} phase consistency')
  plt.legend()
  SaveFig(f'consistency_recall_notrecall_{settings.expreg}')

  pltmin = -0.0018
  pltmax = 0.0045
  StartFig()
  plt.axhline(y=0, color='grey', linestyle='dotted')
  plt.plot(xax, diff_freq, color='blue', label='Recalled - Not-recalled')
  plt.fill_between(xax, diff_conf_low, diff_conf_high, color='blue', alpha=0.4)
  sig_ind = np.argwhere(FDR_sig).ravel().tolist()
  plt.plot(xax, diff_freq, ' ', markevery=sig_ind, marker='*', \
           markersize=10, markeredgecolor='m', markerfacecolor='red',
           label='FDR Sig. weighted paired t-test')
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Phase consistency difference')
  plt.title(f'{settings.reg_short} Phase Consistency SME')
  plt.ylim((pltmin, pltmax))
  plt.legend()
  plt.tight_layout()
  SaveFig(f'consistency_diff_{settings.expreg}')

  #PlotConsistencyFreq(diff_freq, 'Recalled - Not Recalled')
  #plt.savefig('consistency_difference.png')


def RunLowRepPhaseConsistency(sub):
    import numpy as np
    import os
    
    from CMLTools import Settings, SubjectDataFrames, CMLReadDFRow, Locator, \
                         LogDFErr, SetLogger, SpectralAnalysis
    
    try:
        error_suffix = 'lowrep_'+sub
        
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
            morlet_reps=2, time_range=settings.time_range, bin_Hz=settings.bin_Hz,
            buf_ms=settings.buf_ms)
        (rec_results, nrec_results) = sa.PhaseConsistency()

        JSSave(os.path.join(settings.outdir,
            f'lowrep_consistency_{settings.expreg}_{sub}.json'),
            {'rec':rec_results, 'nrec':nrec_results, 'Ns':sa.Ns})
    except Exception as e:
        LogErr(e)
        return False
        
    return True


if run.clusterlowrep:
  ClusterChecked(RunLowRepPhaseConsistency, subs_selected, settings=settings)


if run.plots:
  freqs = settings.freqs
  data = JSLoad(os.path.join(settings.outdir,
      f'lowrep_consistency_{settings.expreg}_{subs_selected[0]}.json'))
  rec_consistency, nrec_consistency, Ns = data['rec'], data['nrec'], data['Ns']
  rec_shape = rec_consistency[0].shape
  nrec_shape = nrec_consistency[0].shape
  xlabels = np.arange(settings.time_range[0], settings.time_range[1]+1, 125)
  xticks = np.linspace(0, rec_shape[1]+1, len(xlabels))
  ylabels = freqs[::3]
  yticks = np.linspace(0, rec_shape[0]-1, len(ylabels))

  rec_list = []
  nrec_list = []
  Ns_list = []
  diff_list = []
  rec_arr = np.zeros(rec_shape)
  nrec_arr = np.zeros(nrec_shape)
  lst_cnt = 0
  sub_cnt = 0
  for sub in subs_selected:
      try:
          data = JSLoad(os.path.join(settings.outdir,
              f'lowrep_consistency_{settings.expreg}_{sub}.json'))
          rec_consistency, nrec_consistency, Ns = data['rec'], data['nrec'], data['Ns']
      except:
          print('File missing for '+sub)
          continue
      if np.any(np.isnan(rec_consistency)) or np.any(np.isnan(nrec_consistency)):
          print('Discarding '+sub+' for nan values.')
          continue
      rec_arr += rec_consistency[0]*Ns[0]
      nrec_arr += nrec_consistency[0]*Ns[0]
      lst_cnt += Ns[0]
      sub_cnt += 1
      nrec_list.append(nrec_consistency[0])
      rec_list.append(rec_consistency[0])
      Ns_list.append(Ns[0])
      diff_list.append(rec_consistency[0]-nrec_consistency[0])
  print(str(sub_cnt)+' subjects loaded.')

  rec_arr /= lst_cnt
  nrec_arr /= lst_cnt
  diff_arr = rec_arr-nrec_arr

  PlotPhaseConsistency(rec_arr, f'{settings.reg_short}, Recalled', \
              f'lowrep_consistency_recalled_phase_consistency_{settings.expreg}')

  PlotPhaseConsistency(nrec_arr, f'{settings.reg_short}, Not Recalled', \
              f'lowrep_consistency_notrecalled_phase_consistency_{settings.expreg}')

  PlotPhaseConsistency(diff_arr, f'{settings.reg_short}, Recalled - Not Recalled', \
              f'lowrep_delta_consistency_phase_consistency_{settings.expreg}')


if run.main:
  print('Ran', exper_list, analysis_region)
  print('Runtime', bench.TotalStr())
  print('Synchronizing outputs on NFS.')
  os.system('sync')

