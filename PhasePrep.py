#!/usr/bin/env python3

from CMLTools import *

settings = Settings()
#settings.exp_list = ['FR1']
settings.exp_list = ['catFR1']
#settings.reg = 'hip'
#settings.reg = 'ltc'
#settings.reg = 'pfc'
settings.reg = 'all'
settings.freqs = np.arange(3, 25, 1)
settings.contact_minimum = 1
settings.list_count_minimum = 25
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
  settings.reg_short = 'All'
else:
  raise ValueError(f'Unknown region specified: {settings.reg}')

bad_list = []
if settings.exp_list == ['FR1']:
  bad_list.append('R1061T') # nan on avg EEG

exp_str = '_'.join(settings.exp_list)
expreg_str = f'{exp_str}_{settings.reg}'

Log = SetLogger(f'preplog_{expreg_str}.txt', datestamp=False)
Log(date=datetime.datetime.now().strftime('%F_%H-%M-%S'))
Log(settings=settings)


# Generate unvalidated subject list

df_exp = DataFramesFor(settings.exp_list)

all_exp_subs = set()
sub_reg_chans = dict()
sub_masks = dict()
sub_session_counts = dict()
sub_list_counts = dict()
sub_no_reg_chans = set()
mismatched = set()
sub_exception_thrown = set()
for row in df_exp.itertuples():
    try:
        all_exp_subs.add(row.subject)
        if row.subject in mismatched or row.subject in sub_no_reg_chans:
            continue
        reader = CMLReadDFRow(row)
        evs = reader.load('task_events')
        enc_evs = evs[evs.type == 'WORD']
        enc_evs = enc_evs[enc_evs.list>0]
        word_list_cnt = len(set(enc_evs.list))
        locmask = Locator(reader).Regions(settings.regions)
        loccnt = sum(locmask)
        if (loccnt < settings.contact_minimum):
            if row.subject in sub_reg_chans:
                mismatched.add(row.subject)
                sub_session_counts.pop(row.subject)
                sub_list_counts.pop(row.subject)
            sub_no_reg_chans.add(row.subject)
            continue
        if row.subject not in sub_reg_chans:
            sub_reg_chans[row.subject] = loccnt
            sub_masks[row.subject] = locmask
            sub_session_counts[row.subject] = 1
            sub_list_counts[row.subject] = word_list_cnt
        else:
            # If electrodes match between sessions, keep subject
            if locmask == sub_masks[row.subject]:
                sub_session_counts[row.subject] += 1
                sub_list_counts[row.subject] += word_list_cnt
            else:
                mismatched.add(row.subject)
                sub_session_counts.pop(row.subject)
                sub_list_counts.pop(row.subject)

    except Exception as e:
        sub_exception_thrown.add(row.subject)
        LogDFException(row, e, 'hipscan')

sub_too_few_lists = {k:v for k,v in sub_list_counts.items() if v<settings.list_count_minimum}
sub_list_counts = {k:v for k,v in sub_list_counts.items() if v>=settings.list_count_minimum}
sub_session_counts = {k:v for k,v in sub_session_counts.items() if k in sub_list_counts}
subs_unvalidated_reg = list(sub_list_counts.keys())
JSSave(f'prep_data_{expreg_str}.json',
    {'unvalidated': subs_unvalidated_reg,
     'list_counts': sub_list_counts,
     'reg_chans': sub_reg_chans,
     'all_exp_subs': list(all_exp_subs),
     'session_counts': sub_session_counts,
     'no_reg_chans': list(sub_no_reg_chans),
     'too_few_lists': sub_too_few_lists,
     'masks': sub_masks,
     'mismatched': list(mismatched),
     'exception_thrown': list(sub_exception_thrown)})

Log('subs_unvalidated_reg:', len(subs_unvalidated_reg), subs_unvalidated_reg)
Log('sub_list_counts:', len(sub_list_counts), sub_list_counts)
Log('sub_session_counts:', len(sub_session_counts), sub_session_counts)
selected_sub_reg_chans = {k:int(v) for k,v in sub_reg_chans.items() if k in subs_unvalidated_reg}
selected_mismatched = [m for m in mismatched if (m not in sub_no_reg_chans)]
Log('selected sub_reg_chans', len(selected_sub_reg_chans), selected_sub_reg_chans)
Log('all sub_reg_chans:', len(sub_reg_chans), sub_reg_chans)
Log(len(all_exp_subs), 'total subs evaluated')
Log(len(sub_exception_thrown), 'subs threw exceptions while loading basic data')
Log(len(sub_no_reg_chans), 'subs with no hip elecs')
Log(len(sub_too_few_lists), 'subs with hip elecs but too few lists')
Log(len(mismatched), 'mismatched montages:', mismatched)
Log(len(subs_unvalidated_reg), 'unvalidated subjects left')

# Validate eeg / pairs

def UpdateListCount(sub, list_counts, num_lists):
    if sub not in list_counts:
        list_counts[sub] = num_lists
    else:
        list_counts[sub] += num_lists

good_list = []
problem_list = []
sr_dict = {}
recall_frac = {}
validated_list_counts = dict()
for sub in subs_unvalidated_reg:
    stats = SubjectStats()
    df_sub = SubjectDataFrames(sub, settings.exp_list)
    locmasks = []

    #print('Subject', sub)
    sess_cnt = 0
    good_sess = 0
    bad_sess = 0
    for row in df_sub.itertuples():
        try:
            #print('session', sess_cnt)
            reader = CMLReadDFRow(row)
            locmask = Locator(reader).Regions(settings.regions)
            locmasks.append(locmask)
            #print(len(locmask))
            evs = reader.load('task_events')
            stats.Add(evs)
            enc_evs = evs[evs.type=='WORD']
            enc_evs = enc_evs[enc_evs.list>0]
            word_list_cnt = len(set(enc_evs.list))
            
            eeg = reader.load_eeg(events=enc_evs, rel_start=0, \
                            rel_stop=100, clean=True)
            sr_dict.setdefault(sub, []).append(float(eeg.samplerate))
            #print(eeg.data.shape)
            pairs = reader.load('pairs')
            #print(pairs.shape)
            if pairs.shape[0] != eeg.data.shape[1]:
                #print('Trying reload with pairs')
                eeg = reader.load_eeg(events=enc_evs, rel_start=0, \
                            rel_stop=100, scheme=pairs, clean=True)
                if pairs.shape[0] != eeg.data.shape[1]:
                    #print('Mismatched eeg / pairs!')
                    bad_sess += 1
                else:
                    #print('Salvaged')
                    good_sess += 1
                    UpdateListCount(sub, validated_list_counts, word_list_cnt)
            else:
                #print('Valid session for eeg / pairs')
                good_sess += 1
                UpdateListCount(sub, validated_list_counts, word_list_cnt)
        except Exception as e:
            LogDFException(row, e, 'validate')
            bad_sess += 1
        sess_cnt += 1

    recall_frac[sub] = stats.RecallFraction()
    #print()
    if good_sess > 0:
        good_list.append(sub)
    else:
        bad_list.append(sub)
    if bad_sess > 0:
        problem_list.append(sub)

subs_reg = good_list
sub_list_counts = {k:int(v) for k,v in validated_list_counts.items() if v>=settings.list_count_minimum and k not in bad_list}
good_sub_sr = {k:v for k,v in sr_dict.items() if k in good_list}
good_recall_frac = {k:float(v) for k,v in recall_frac.items() if k in good_list}
JSSave(f'subject_list_{expreg_str}.json',
    {'subs': list(sub_list_counts),
     'sub_list_counts': sub_list_counts,
     'chan_counts': selected_sub_reg_chans,
     'samplerates': good_sub_sr,
     'recall_fraction': good_recall_frac})

Log('Subjects evaluated:', len(subs_unvalidated_reg), subs_unvalidated_reg)
Log('Good list:', len(good_list), good_list)
Log('Bad list:', len(bad_list), bad_list)
Log('Problem list:', len(problem_list), problem_list)
Log('Validated sub list counts:', len(validated_list_counts), validated_list_counts)

