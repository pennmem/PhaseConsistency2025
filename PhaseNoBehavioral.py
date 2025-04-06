#!/usr/bin/env python3

from CMLTools import *

run = RunOpts(__name__, ['setup', 'params', 'cluster', 'plots'])

exper_list = ['FR1']
#exper_list = ['catFR1']
#analysis_region = 'hip'
#analysis_region = 'ltc'
#analysis_region = 'pfc'
analysis_region = 'all'
settings_file = f'phase_no_behav_{"_".join(exper_list)}_{analysis_region}.json'
if run.setup:
  settings = Settings()
  settings.freqs = np.arange(3, 25, 1)
  settings.exp_list = exper_list
  settings.time_range = (-750, 750)
  settings.bin_Hz = 250
  #settings.spos_ranges = [[i] for i in range(1,13)]
  settings.spos_ranges = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
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

  settings.outdir = f'phasenobeh_{settings.expreg}'
  settings.logfile = os.path.join(settings.logdir,
      settings.outdir+'.txt')

  settings.Save(settings_file)

  os.makedirs(settings.outdir, exist_ok=True)
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

def RunPhaseConsistencyNoBeh(sub):
  import numpy as np
  import os
  
  from CMLTools import Settings, SubjectDataFrames, CMLReadDFRow, Locator, \
             LogDFException, LogException, SpectralAnalysis
  
  try:
    error_suffix = 'nobeh_'+sub
    
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
        LogDFException(row, e, error_suffix)
      sess_cnt += 1
    mask_array = np.zeros(len(df_sub), dtype=bool)
    mask_array[valid_sess] = 1
    df_sub = df_sub[mask_array]

    sa = SpectralAnalysis(settings.freqs, dfs=df_sub, elec_masks=locmasks,
        time_range=settings.time_range, bin_Hz=settings.bin_Hz,
        split_recall=False)
    results = sa.PhaseConsistencySerPos(settings.spos_ranges)

    JSSave(os.path.join(settings.outdir,
        f'consistency_nobeh_{settings.expreg}_{sub}.json'),
        {'res':results, 'Ns':sa.Ns})
    JSSave(os.path.join(settings.outdir,
        f'spos_recall_{settings.expreg}_{sub}.json'),
        {'sposarr':sa.sposarr, 'list_count':sa.list_count})
  except Exception as e:
    LogException(e, error_suffix)
    return False
    
  return True


if run.cluster:
  ClusterChecked(RunPhaseConsistencyNoBeh, subs_selected, settings=settings)


if run.plots:
  data = JSLoad(os.path.join(settings.outdir,
          f'consistency_nobeh_{settings.expreg}_{subs_selected[0]}.json'))
  freqs = settings.freqs
  spos_ranges = settings.spos_ranges
  res_consistency = data['res']
  Ns = data['Ns']
  spos_len = len(spos_ranges)
  spos_labels = [str(s[0])+'-'+str(s[-1]) if len(s)>1 else str(s[0])
      for s in spos_ranges]
  res_shape = res_consistency[0][0].shape
  xlabels = np.arange(settings.time_range[0], settings.time_range[1]+1, 125)
  xticks = np.linspace(0, res_consistency[0][0].shape[1]+1, len(xlabels))
  ylabels = freqs[::3]
  yticks = np.linspace(0, res_consistency[0][0].shape[0]-1, len(ylabels))

  for spos in range(spos_len):
    res_arr = np.zeros(res_shape)
    sub_cnt = 0
    for sub in subs_selected:
      try:
        data = JSLoad(os.path.join(settings.outdir,
                f'consistency_nobeh_{settings.expreg}_{sub}.json'))
        spos_ranges = settings.spos_ranges
        res_consistency = data['res']
      except:
        print('File missing for '+sub)
        continue
      if np.any(np.isnan(res_consistency)):
        print('Discarding '+sub+' for nan values.')
        continue
      res_arr += res_consistency[0][spos]
      sub_cnt += 1
    print(str(sub_cnt)+' subjects loaded.')

    res_arr /= sub_cnt

    PlotPhaseConsistency(res_arr,
        f'Phase Consistency, Ser. Pos. {spos_labels[spos]}', \
          f'consistency_nobeh_spos_{settings.expreg}_{spos_labels[spos]}')


if run.plots:
  print(ylabels, yticks)
  data = JSLoad(os.path.join(settings.outdir,
          f'consistency_nobeh_{settings.expreg}_{subs_selected[0]}.json'))
  freqs = settings.freqs
  spos_ranges = settings.spos_ranges
  res_consistency = data['res']
  Ns = data['Ns']
  spos_len = len(spos_ranges)
  spos_labels = [str(s[0])+'-'+str(s[-1]) if len(s)>1 else str(s[0])
      for s in spos_ranges]
  res_shape = res_consistency[0][0].shape

  t_arr = []
  p_arr = []
  pre_spos_arr = []
  post_spos_arr = []
  pre_spos_sub_freq_arr = []
  post_spos_sub_freq_arr = []
  pre_all_arr = []
  post_all_arr = []
  last_sub_cnt = -1
  for spos in range(spos_len):
    res_arr = np.zeros(res_shape)
    sub_cnt = 0
    Ns = []
    pre_exp_arr = []
    post_exp_arr = []
    pre_exp_sub_freq_arr = []
    post_exp_sub_freq_arr = []
    for sub in subs_selected:
      try:
        data = JSLoad(os.path.join(settings.outdir,
                f'consistency_nobeh_{settings.expreg}_{sub}.json'))
        spos_ranges = settings.spos_ranges
        res_consistency = data['res']
        Ns_sub = data['Ns']
      except:
        print('File missing for '+sub)
        continue
      if np.any(np.isnan(res_consistency)):
        print('Discarding '+sub+' for nan values.')
        continue
      res_sub = res_consistency[0][spos]
      res_arr += res_consistency[0][spos]

      res_sub = res_consistency[0][spos]
      FreqLow = np.argwhere(freqs==3)[0][0]
      FreqHigh = np.argwhere(freqs==24)[0][0]
      pre_exposure = np.mean(res_sub[FreqLow:FreqHigh+1,
          0:int((-375-(-750))/4+0.5)+1])
      post_exposure = np.mean(res_sub[FreqLow:FreqHigh+1,
          int((750+125)/4+0.5):int((750+500)/4+0.5)+1])
      pre_exp_arr.append(pre_exposure)
      post_exp_arr.append(post_exposure)
      pre_exp_sub_freq_arr.append(np.mean(res_sub[:, 0:int((-375-(-750))/4+0.5)+1], axis=1))
      post_exp_sub_freq_arr.append(np.mean(res_sub[:, int((750+125)/4+0.5):int((750+500)/4+0.5)+1], axis=1))
      Ns.append(Ns_sub[0])

      sub_cnt += 1
    
    weights = np.array(Ns)/np.mean(Ns)
    pre_spos_arr.append(np.mean(np.array(pre_exp_arr)*weights))
    post_spos_arr.append(np.mean(np.array(post_exp_arr)*weights))
    pre_all_arr.append(pre_exp_arr)
    post_all_arr.append(post_exp_arr)
    pre_spos_sub_freq_arr.append(pre_exp_sub_freq_arr)
    post_spos_sub_freq_arr.append(post_exp_sub_freq_arr)
    
    if last_sub_cnt < 0:
      last_sub_cnt = sub_cnt
    else:
      if sub_cnt != last_sub_cnt:
        print('Mismatched subject count,', last_sub_cnt, '!=', sub_cnt,
            'for spos', spos)

    res_arr /= sub_cnt
    #print(post_exp_arr, pre_exp_arr, Ns)
    t, p = WeightedNPairedTTest(post_exp_arr, pre_exp_arr, Ns)
    t_arr.append(t)
    p_arr.append(p)
    print('For spos', spos, 't =', t, 'p =', p)

  pre_spos_sub_freq_arr = np.array(pre_spos_sub_freq_arr)
  post_spos_sub_freq_arr = np.array(post_spos_sub_freq_arr)
  print(str(last_sub_cnt)+' subjects analyzed.')


if run.plots:
  spos_weights = np.array(Ns)/np.mean(Ns)
  desc_pre = statsmodels.stats.weightstats.DescrStatsW(
      np.array(pre_all_arr).T, spos_weights, ddof=1)
  desc_post = statsmodels.stats.weightstats.DescrStatsW(
      np.array(post_all_arr).T, spos_weights, ddof=1)
  pre_conf_low, pre_conf_high = desc_pre.tconfint_mean()
  post_conf_low, post_conf_high = desc_post.tconfint_mean()

  # Benjamini-Hochberg FDR correction
  res = statsmodels.stats.multitest.multipletests(p_arr, 0.05,
      method='fdr_bh')
  FDR_sig = res[0]
  pval_corr = res[1]
  print('False discovery rate corrected:', FDR_sig)
  print('Corrected pvals:', pval_corr)
  print('Number significant:', np.sum(FDR_sig))

  x_spos = ['-'.join(str(s) for s in r) for r in settings.spos_ranges]

  StartFig()
  sig_ind = np.argwhere(FDR_sig).ravel().tolist()
  plt.plot(x_spos, t_arr, markevery=sig_ind, marker='*', \
       markersize=10, markeredgecolor='r', markerfacecolor='r')
  plt.xlabel('Serial Position')
  plt.ylabel('weighted paired t')
  plt.title('Post-word period vs. pre-word period')
  SaveFig(f'spos_tplot_{settings.expreg}')

  StartFig()
  plt.plot(x_spos, pre_spos_arr, color='red', label='Pre word exposure')
  plt.fill_between(x_spos, pre_conf_low, pre_conf_high, color='red',
      alpha=0.4)
  plt.plot(x_spos, post_spos_arr, color='blue', label='Post word exposure')
  plt.fill_between(x_spos, post_conf_low, post_conf_high, color='blue',
      alpha=0.4)
  plt.xlabel('Serial Position')
  plt.ylabel('Phase Consistency')
  plt.legend()
  SaveFig(f'spos_phase_consistency_{settings.expreg}')

  pre_consistency_subs = np.array(pre_all_arr).T
  post_consistency_subs = np.array(post_all_arr).T
  total_lists = np.sum(Ns)
  avg_pre_con = np.zeros(len(settings.spos_ranges))
  avg_post_con = np.zeros(len(settings.spos_ranges))
  avg_phase_con_diff = np.zeros(len(settings.spos_ranges))
  avg_correct = np.zeros(len(settings.spos_ranges))
  num_plots = 0
  plt_cnt = 0
  spos_labs = [str(s) for s in settings.spos_ranges]
  for sub,pre_consistency,post_consistency,N in zip(subs_selected,
      pre_consistency_subs, post_consistency_subs, Ns):
    try:
      data = JSLoad(os.path.join(settings.outdir,
          f'spos_recall_{settings.expreg}_{sub}.json'))
      (sposarr, list_count) = data['sposarr'], data['list_count']
    except Exception as e:
      print(f'File missing for {sub}, {e}')
      continue

    for r in range(len(settings.spos_ranges)):
      avg_correct[r] += N*np.mean(np.array(sposarr)[np.array(
          settings.spos_ranges[r])-1])/list_count

    avg_pre_con += N*pre_consistency
    avg_post_con += N*post_consistency
    avg_phase_con_diff += N*(post_consistency-pre_consistency)

    if plt_cnt < num_plots:
      plt_cnt += 1
      plt.figure(figsize=(12,5))
      plt.subplot(121)
      plt.plot(np.arange(len(sposarr)), sposarr/list_count)
      plt.xticks(np.arange(len(sposarr)))
      plt.xlabel('Serial Position')
      plt.ylabel('Fraction Recalled')
      plt.subplot(122)
      plt.plot(spos_labs, post_consistency)
      plt.plot(settings.spos_ranges, post_consistency)
      #plt.xticks(settings.spos_ranges)
      plt.xlabel('Serial Position')
      plt.ylabel('Phase Consistency (Post Word)')

  avg_correct /= total_lists
  avg_pre_con /= total_lists
  avg_post_con /= total_lists
  avg_phase_con_diff /= total_lists

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4.5))
  fig.subplots_adjust(wspace=0.25)
  ax1.plot(x_spos, pre_spos_arr, color='red', label='Pre word exposure')
  ax1.fill_between(x_spos, pre_conf_low, pre_conf_high, color='red',
      alpha=0.4)
  ax1.plot(x_spos, post_spos_arr, color='blue', label='Post word exposure')
  ax1.plot(x_spos, post_spos_arr, ' ', color='g', label='Post/pre sig. after FDR cor.',
      markevery=sig_ind, marker='*', markersize=10, markeredgecolor='r',
      markerfacecolor='r')
  ax1.fill_between(x_spos, post_conf_low, post_conf_high, color='blue',
      alpha=0.4)
  ax1.set_xlabel('Serial Position')
  ax1.set_ylabel('Phase Consistency')
  ax1.legend()
  print(['-'.join(str(s) for s in r) for r in settings.spos_ranges])
  ax1.set_xticks(['-'.join(str(s) for s in r) for r in settings.spos_ranges])
  ax1.set_xlabel('Serial Position')
  ax1.set_ylabel('Average Phase Consistency')
  ax1.set_title('Phase consistency over serial position')

  ax2b = ax2.twinx()
  ax2.set_title('Post-word consistency and avg. recall by serial pos.')
  ax2.plot(x_spos, avg_correct, label='Fraction Recalled')\
  #ax2.set_xticks(settings.spos_ranges)
  ax2.set_xlabel('Serial Position')
  ax2.set_ylabel('Average Fraction Recalled', color='blue')
  #ax2.set_ylim(round(np.min(avg_correct)*0.7-0.05,1), round(np.max(avg_correct)+0.05,1))

#  ax2b.plot(x_spos, t_arr, '--', color='g', label='Weighted paired t')
#  ax2b.plot(x_spos, t_arr, ' ', color='g', label='Signif. after FDR cor.',
#      markevery=sig_ind, marker='*', markersize=10, markeredgecolor='r',
#      markerfacecolor='r')
#  ax2b.set_xlabel('Serial Position')
#  ax2b.set_ylabel('weighted paired t', color='green')
#  ax2b.set_ylim(round(np.min(t_arr)*0.95-0.05,1),
#      round(np.max(t_arr)*1.05+0.05,1))
  ax2b.plot(x_spos, post_spos_arr, '--', color='g', label='Post word phase consistency')
  #ax2b.plot(x_spos, post_spos_arr, ' ', color='g', label='Post/pre sig. after FDR cor.',
  #    markevery=sig_ind, marker='*', markersize=10, markeredgecolor='r',
  #    markerfacecolor='r')
  ax2b.set_xlabel('Serial Position')
  ax2b.set_ylabel('Phase consistency', color='green')
  ax2b.set_ylim(round(np.min(post_spos_arr)*0.99,5),
      round(np.max(post_spos_arr)*1.01,5))
  lines2, labels2 = ax2.get_legend_handles_labels()
  lines2b, labels2b = ax2b.get_legend_handles_labels()
  ax2.legend(lines2 + lines2b, labels2 + labels2b, loc=0)
  SaveFig(f'spos_avg_consistency_recall_and_phase_{settings.expreg}')


  fig = StartFig()
  ax1 = fig.add_subplot(111)
  ax1.plot(x_spos, pre_spos_arr, color='red', label='Pre word exposure')
  ax1.fill_between(x_spos, pre_conf_low, pre_conf_high, color='red',
      alpha=0.4)
  ax1.plot(x_spos, post_spos_arr, color='blue', label='Post word exposure')
  ax1.plot(x_spos, post_spos_arr, ' ', color='g', label='Post/pre sig. after FDR cor.',
      markevery=sig_ind, marker='*', markersize=10, markeredgecolor='r',
      markerfacecolor='r')
  ax1.fill_between(x_spos, post_conf_low, post_conf_high, color='blue',
      alpha=0.4)
  ax1.set_xlabel('Serial Position')
  ax1.set_ylabel('Phase Consistency')
  ax1.legend()
  ax1.set_xticks(['-'.join(str(s) for s in r) for r in settings.spos_ranges])
  ax1.set_xlabel('Serial Position')
  ax1.set_ylabel('Average Phase Consistency')
  ax1.set_title('Phase consistency over serial position')
  plt.tight_layout()
  SaveFig(f'spos_avg_consistency_{settings.expreg}')

  fig = StartFig()
  ax2 = fig.add_subplot(111)
  ax2b = ax2.twinx()
  ax2.set_title('Post-word consistency and avg. recall by serial pos.')
  ax2.plot(x_spos, avg_correct, label='Fraction Recalled')\
  #ax2.set_xticks(settings.spos_ranges)
  ax2.set_xlabel('Serial Position')
  ax2.set_ylabel('Average Fraction Recalled', color='blue')
  #ax2.set_ylim(round(np.min(avg_correct)*0.7-0.05,1), round(np.max(avg_correct)+0.05,1))

#  ax2b.plot(x_spos, t_arr, '--', color='g', label='Weighted paired t')
#  ax2b.plot(x_spos, t_arr, ' ', color='g', label='Signif. after FDR cor.',
#      markevery=sig_ind, marker='*', markersize=10, markeredgecolor='r',
#      markerfacecolor='r')
#  ax2b.set_xlabel('Serial Position')
#  ax2b.set_ylabel('weighted paired t', color='green')
#  ax2b.set_ylim(round(np.min(t_arr)*0.95-0.05,1),
#      round(np.max(t_arr)*1.05+0.05,1))
  ax2b.plot(x_spos, post_spos_arr, '--', color='g', label='Post word phase consistency')
  #ax2b.plot(x_spos, post_spos_arr, ' ', color='g', label='Post/pre sig. after FDR cor.',
  #    markevery=sig_ind, marker='*', markersize=10, markeredgecolor='r',
  #    markerfacecolor='r')
  ax2b.set_xlabel('Serial Position')
  ax2b.set_ylabel('Phase consistency', color='green')
  ax2b.set_ylim(round(np.min(post_spos_arr)*0.99,5),
      round(np.max(post_spos_arr)*1.01,5))
  lines2, labels2 = ax2.get_legend_handles_labels()
  lines2b, labels2b = ax2b.get_legend_handles_labels()
  ax2.legend(lines2 + lines2b, labels2 + labels2b, loc=0)
  plt.tight_layout()
  SaveFig(f'spos_avg_recall_and_phase_{settings.expreg}')

# StartFig(figsize=(12,5))
# plt.subplot(121)
# plt.plot(x_spos, pre_spos_arr, color='red', label='Pre word exposure')
# plt.fill_between(x_spos, pre_conf_low, pre_conf_high, color='red', alpha=0.4)
# plt.plot(x_spos, post_spos_arr, color='blue', label='Post word exposure')
# plt.fill_between(x_spos, post_conf_low, post_conf_high, color='blue', alpha=0.4)
# plt.xlabel('Serial Position')
# plt.ylabel('Phase Consistency')
# plt.legend()
# plt.xticks(settings.spos_ranges)
# plt.xlabel('Serial Position')
# plt.ylabel('Average Phase Consistency')
# #plt.subplot(122)
# fig, ax1 = plt.subplots()
# plt.title('Post-word period vs. pre-word period')
# ax1.plot(settings.spos_ranges, avg_correct)
# ax1.set_xticks(settings.spos_ranges)
# ax1.set_xlabel('Serial Position')
# ax1.set_ylabel('Average Fraction Recalled', color='blue')
# ax2 = ax1.twinx()
# ax2.plot(x_spos, t_arr, '--', color='g', markevery=sig_ind, marker='*', \
#      markersize=10, markeredgecolor='r', markerfacecolor='r')
# ax2.set_xlabel('Serial Position')
# ax2.set_ylabel('weighted paired t', color='green')
# ax2.set_ylim(-4, -8.5)
# SaveFig('spos_avg_recall_and_phase')


if run.plots:
  # Loop over every subject and gather correlation R for each.

  sub_cnt = 0
  R_list = []
  m_list = []
  b_list = []
  R_sub_freq_list = []
  m_sub_freq_list = []
  b_sub_freq_list = []
  for si,(sub,pre_consistency,post_consistency,N) in enumerate(zip(subs_selected, pre_consistency_subs, post_consistency_subs, Ns)):
    try:
      data = JSLoad(os.path.join(settings.outdir,
              f'spos_recall_{settings.expreg}_{sub}.json'))
      sposarr = data['sposarr']
      list_count = data['list_count']
    except Exception as e:
      print(f'File missing for {sub}, {e}')
      continue

    #frac_recalled = sposarr/list_count
    frac_recalled = np.zeros(len(settings.spos_ranges))
    for r in range(len(settings.spos_ranges)):
      frac_recalled[r] = np.sum(np.array(sposarr)[np.array(
          settings.spos_ranges[r])-1])/list_count

    delta_phase_con = post_consistency - pre_consistency
    
    m,b,R,p,err = scipy.stats.linregress(frac_recalled, delta_phase_con)
    R_list.append(R)
    m_list.append(m)
    b_list.append(b)
    #print(m,b,R,p,err)
    sub_cnt += 1
    if sub_cnt < 5:
      StartFig()
      plt.plot(frac_recalled, delta_phase_con, 'o')
      plt.xlabel('Fraction Recalled for Ser. Pos.')
      plt.ylabel('Post-Pre Phase Consistency difference')
      xarr = np.linspace(np.min(frac_recalled), np.max(frac_recalled), 100)
      yarr = m*xarr+b
      pltlab = 'R={:.3f}, p={:.3f}'.format(R, p)
      plt.plot(xarr, yarr, color='red', label=pltlab)
      plt.legend()
      SaveFig(f'test_fig_{settings.expreg}')

    R_freq_list = []
    m_freq_list = []
    b_freq_list = []
    for fi,f in enumerate(settings.freqs):
      pre_spos_arr = pre_spos_sub_freq_arr[:, si, fi]
      post_spos_arr = post_spos_sub_freq_arr[:, si, fi]
      delta_spos_arr = post_spos_arr - pre_spos_arr

      m,b,R,p,err = scipy.stats.linregress(frac_recalled, delta_spos_arr)
      R_freq_list.append(R)
      m_freq_list.append(m)
      b_freq_list.append(b)
    
    R_sub_freq_list.append(R_freq_list)
    m_sub_freq_list.append(m_freq_list)
    b_sub_freq_list.append(b_freq_list)

  R_freq_sub_arr = np.array(R_sub_freq_list).T

  m,b,R,_,err = scipy.stats.linregress(avg_correct, avg_phase_con_diff)
  print(m,b,R,p,err)
  StartFig()
  plt.plot(avg_correct, avg_phase_con_diff, 'o')
  plt.xlabel('Fraction Recalled for Ser. Pos.')
  plt.ylabel('Post-Pre Phase Consistency Difference')
  xarr = np.linspace(np.min(avg_correct), np.max(avg_correct), 100)
  yarr = m*xarr+b
  #pltlab = 'R={:.3f}, Across Subj. p={:.3f}'.format(R, p)
  pltlab = 'R={:.3f}'.format(R, p)
  plt.plot(xarr, yarr, color='red', label=pltlab)
  plt.title('Each serial position, avgd. over subjects')
  plt.legend()
  plt.tight_layout()
  SaveFig(f'spos_phase_consistency_vs_recall_{settings.expreg}')
  m_avg = m
  b_avg = b

  StartFig()
  for m,b in zip(m_list, b_list):
    xarr = np.linspace(0.15, 0.8, 100)
    yarr = m*xarr+b
    plt.plot(xarr, yarr, color='red', label=pltlab)
  print('median m: ', m, ', median b: ', b, sep='')
  xarr = np.linspace(0.15, 0.8, 100)
  yarr = m_avg*xarr+b_avg
  plt.plot(xarr, yarr, color='black', label=pltlab)
  SaveFig('test_fig')


  t_R, p_R = WeightedN1SampTTest(R_list, 0, Ns)
  #t_R, p_R = scipy.stats.ttest_1samp(R_list, 0)
  print('avg R: ', np.mean(R_list), ', t: ', t_R, ', p: ', p_R, sep='')

  direction = 'R>0' if t_R > 0 else 'R<0'
  import matplotlib.ticker as mticker
  form = mticker.ScalarFormatter(useOffset=False, useMathText=True)
  form.set_scientific(True) 
  pstr = f'{p_R:.2}'
  # '2 \\times 10^{5}'
  #pstr = f'${form.format_data(p_R)}$'
  mu, SD, sem = WeightedMeanSDSem(R_list, Ns)

  barfactor = 1.9599639845400542
  StartFig()
  n,bins,patches = plt.hist(R_list, bins=14,
      label=f'p = {pstr} for {direction}')
  plt.errorbar([mu], [np.max(n)/25.0], fmt='+', xerr=[sem*barfactor],
      label='95% confidence interval',
      #label='$\pm$ one std. err. of mean',
      markeredgewidth=4, elinewidth=4, capthick=4, capsize=10, color='black')
  plt.xlabel('R, phase consistency post-pre. and ser. pos. recall rate')
  plt.ylabel('Subject count')
  plt.xlim(-1, 1)
  #plt.title('Binned recall and phase consistency correl. across subjects    ')
  plt.legend(loc='upper left')
  SaveFig(f'Rhist_postpre_serpos_binned_{settings.expreg}')

  mu_freq_list = []
  sem_freq_list = []
  p_R_list = []
  for fi,f in enumerate(settings.freqs):
    t_R, p_R = WeightedN1SampTTest(R_freq_sub_arr[fi], 0, Ns)
    p_R_list.append(p_R)
    #t_R, p_R = scipy.stats.ttest_1samp(R_list, 0)
    print('f: ', f, 'Hz, avg R: ', np.mean(R_freq_sub_arr[fi]), ', t: ', t_R, ', p: ', p_R, sep='')

    direction = 'R>0' if t_R > 0 else 'R<0'
    import matplotlib.ticker as mticker
    form = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    form.set_scientific(True) 
    pstr = f'{p_R:.2}'
    # '2 \\times 10^{5}'
    #pstr = f'${form.format_data(p_R)}$'
    mu, SD, sem = WeightedMeanSDSem(R_freq_sub_arr[fi], Ns)
    mu_freq_list.append(mu)
    sem_freq_list.append(sem)

    StartFig()
    n,bins,patches = plt.hist(R_freq_sub_arr[fi], bins=14,
        label=f'p = {pstr} for {direction}')
    plt.errorbar([mu_freq_list[fi]], [np.max(n)/25.0], fmt='+',
        xerr=[sem_freq_list[fi]*barfactor],
        label='95% confidence interval',
        #label='$\pm$ one std. err. of mean',
        markeredgewidth=4, elinewidth=4, capthick=4, capsize=10, color='black')
    plt.xlabel(f'{f}Hz R, phase consistency post-pre. and ser. pos. recall rate')
    plt.ylabel('Subject count')
    plt.xlim(-1, 1)
    #plt.title('Binned recall and phase consistency correl. across subjects    ')
    plt.legend(loc='upper left')
    SaveFig(f'Rhist_postpre_serpos_binned_{settings.expreg}_{int(f):02}Hz')

  res = statsmodels.stats.multitest.multipletests(p_R_list, 0.05,
      method='fdr_bh')
  FDR_sig = res[0]
  pval_corr = res[1]

  StartFig()
  sig_ind = np.argwhere(FDR_sig).ravel().tolist()
  mu_freq_arr = np.array(mu_freq_list)
  plt.plot(settings.freqs, mu_freq_arr, color='blue',
      label='Mean serial position group R', markevery=sig_ind, marker='*',
      markersize=10, markeredgecolor='r', markerfacecolor='r')
  barfactor = 1.9599639845400542
  conf_freq_arr = np.array(sem_freq_list)*barfactor
  plt.fill_between(settings.freqs, mu_freq_arr-conf_freq_arr,
      mu_freq_arr+conf_freq_arr, label='95% confidence interval',
      color='blue', alpha=0.4)
  plt.axhline(y=0, color='grey', linestyle='dotted')
  plt.xlabel('Frequency')
  plt.ylabel('Subject mean R, phase consistency vs recall')
  plt.legend()
  SaveFig(f'Rhist_freq_pattern_{settings.expreg}')

if run.plots:
  import seaborn
  StartFig(figsize=(6,6))
  print('mu = ', mu, ', SD = ', SD, ', sem = ', sem, sep='')
  #plt.bar([-1,0,1], [0, mu ,0], yerr=[np.nan, sem, np.nan],
  #    error_kw={'elinewidth':3, 'capthick':3, 'capsize':12},
  #    label=f'p = {pstr}\n for {direction}')
  np.random.seed(0)
  xvals = (np.random.random(len(R_list))-0.5)*2*0.4
  #plt.plot(xvals, R_list, '.', color='#20202080')
  seaborn.swarmplot(data=R_list, size=6, color='grey')
  plt.errorbar([0], [mu], fmt=' ', yerr=[sem],
         elinewidth=4, capthick=4, capsize=16, color='blue')
  plt.plot([0], [mu], '+', markersize=10, markeredgewidth=2, color='blue',
       label=f'p = {pstr}\n for {direction}')
  plt.xticks([])
  plt.xlim(-0.1, 0.17)
  #plt.ylim(bottom=0)
  plt.ylabel('R, post-pre phase consistency diff. vs. recall')
  plt.title('Binned recall and phase consistency correl. across subjects    ')
  plt.legend(loc='upper right')
  SaveFig(f'R_barscatter_postpre_recall_{settings.expreg}')
  def WeightedMeanSDSem(a, Ns):
    '''For 1D or 2D a, NS of same size, returns mu, SD, sem of a, all
     weighted by Ns.'''
    norm_w_a = np.array(Ns)/np.mean(Ns)

    desc = statsmodels.stats.weightstats.DescrStatsW(a, norm_w_a, ddof=1)
    return desc.mean, desc.std, desc.std_mean

  #print('\n'.join(str(s) for s in zip(avg_correct, avg_phase_con)))
  m,b,R,p,err = scipy.stats.linregress(avg_correct, t_arr)
  print(m,b,R,p,err)
  StartFig()
  plt.plot(avg_correct, t_arr, 'o')
  plt.xlabel('Fraction Recalled for Ser. Pos.')
  xarr = np.linspace(np.min(avg_correct), np.max(avg_correct), 100)
  yarr = m*xarr+b
  #pltlab = 'R={:.3f}, p={:.3f}'.format(R, p)  # bad p, not independent.
  pltlab = 'R={:.3f}'.format(R, p)
  plt.plot(xarr, yarr, color='red', label=pltlab)
  plt.ylabel('weighted paired t')
  plt.title('Post-word period vs. pre-word period')
  plt.legend()
  SaveFig(f'spos_t_vs_recall_{settings.expreg}')

if run.main:
  print('Ran', exper_list, analysis_region)
  print('Runtime', bench.TotalStr())
  print('Synchronizing outputs on NFS.')
  os.system('sync')

