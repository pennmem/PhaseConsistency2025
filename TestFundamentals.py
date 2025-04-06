#!/usr/bin/env python3

%matplotlib inline
from CMLTools import *
settings = Settings()
settings.freqs = np.arange(3, 25, 1)
settings.exp_list = ['FR1']
settings.time_range = (-750, 750)
settings.bin_Hz = 250
settings.spos_ranges = [[i] for i in range(1,13)]
settings.Save('test_fundamentals.json')

print('Frequencies:', settings.freqs)
print('Selected:', settings.exp_list)

sub_list_data = JSLoad(f'subject_list_{settings.expreg}.json')
subs_selected = sub_list_data['subs']
sub_list_counts = sub_list_data['sub_list_counts']

def OneSigma(circ_disp, N):
  scaled_sigma = np.sqrt(circ_disp/(N-1))
  if scaled_sigma >= 1:
    return np.pi/2
  retval = np.arcsin(scaled_sigma)*np.sqrt(N)
#  if retval > np.pi/2:
#    return np.pi/2
  #print('before', retval)
  #retval = 1/(1/retval + np.pi**2/(2*np.sqrt(N)))
  #print('after', retval)
  return retval

def PRayleigh(r, N):
  '''Adapted from Rizzuto's PhasePACK in eeg_toolbox.
     Also found in Greenwood and Durand, 1955, taking only through the n^-2 terms.'''
  Z = N*r*r
  return np.exp(-Z) * (1 + (2*Z - Z*Z) / (4*N) - (24*Z - 132*Z*Z + 76*Z**3 - 9* Z**4) / (288*N*N));

def OldPhaseConsistency(circ_disp, N):
  '''0 to 1, where 0 is the 1-sigma phase spread over half or more of the
     circle, and 1 is all points in one location.  0.5 is the 1-sigma
     spread occupying a 90-degree arc centered at the mean value, 0.75 is
     spread across a 45-degree arc, and so on.'''
  return 1 - OneSigma(circ_disp, N)/(np.pi/2)

def PolarHist(ang_arr, file, num_bins=None, title=None):
  autobin = False
  if num_bins is None:
    autobin = True
    num_bins = 2**int(np.floor(np.log2(len(ang_arr)/9.99999)))
    num_bins = min(max(num_bins, 16), 128)
  
  while True:
    h, bins = np.histogram(np.mod(ang_arr, 2*np.pi), num_bins, (0, 2*np.pi))
    if num_bins < 128 and (np.max(h) / len(ang_arr)) >= 0.20:
      num_bins *= 2
      continue
    bins = (bins[:num_bins]+bins[1:])/2
    break

  print(num_bins)
  hstr = ', '.join(str(e) for e in h)
  print(f'[{hstr}]')

  StartFig()
  ax = plt.subplot(111, projection='polar')
  bars = ax.bar(bins, h, width=2*np.pi/num_bins)
  if title is not None:
    plt.title(title)
  for b in bars:
    b.set_alpha(0.7)
  SaveFig(file)

  
def PhaseConsistencyTest(arr, N):
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

N = 1600
sigma = (np.pi/2)*1.38
sigma = (np.pi/2)*1.365
v = []
pc_arr = []
r_arr = []
e_arr = []
z_arr = []
zs_arr = []
pr_arr = []
for i in range(100):
  a = np.random.normal(np.pi, sigma, N)
  if i<2:
    #print(','.join(str(s) for s in a), CircularDispersion([np.cos(x) + 1j*np.sin(x) for x in a]))
    print(np.min(a), np.max(a))
    print(np.std(a, ddof=1))
    PolarHist(a, 'polar_hist')
  p = [np.cos(x) + 1j*np.sin(x) for x in a]
  r = np.abs(np.sum(p))/len(a)
  r_arr.append(r)
  z = N*r*r
  z_arr.append(z)
  zs = (z-1)/(N-1)
#  zs_arr.append(zs)
  zs_arr.append(PhaseConsistencyTest(p, N))
  pr = PRayleigh(r, N)
  pr_arr.append(pr)
  circ_disp = CircularDispersion(p)
  e = np.exp(-circ_disp)
  e_arr.append(e)
  onesig = OneSigma(circ_disp, N)
  pc = OldPhaseConsistency(circ_disp, N)
  #onesig = np.arcsin(np.sqrt(circ_disp))
  #v.append(1/(circ_disp+1))
  v.append(onesig)
  pc_arr.append(pc)
#print(circ_disp, PhaseSpread(circ_disp, N))
#print(1/(circ_disp+1))
print(np.mean(v), sigma, np.mean(pc_arr))
#print('v',v)
#print('r_arr',r_arr)
print('ravg', np.mean(r_arr))
#print('eavg,', np.mean(e_arr), 'e_arr', e_arr)
print('z_avg:', np.mean(z_arr))
#print('z_arr:', z_arr)
print('pr_avg:', np.mean(pr_arr))
#print('pr_arr:', pr_arr)
print('zs_avg', np.mean(zs_arr))
print('np.std(zs_arr):', np.std(zs_arr, ddof=1))
print('zs_arr < 1:', np.sum(np.array(zs_arr)<1)/len(zs_arr))
print('zs_arr > 2:', np.sum(np.array(zs_arr)>2)/len(zs_arr))
#print('zs_arr', zs_arr)
N = 1600
sigma = (np.pi/2)*1.365
zs1 = 0
while zs1 < 0.0099 or zs1 > 0.0101:
  a1 = np.random.normal(np.pi, sigma, N)%(2*np.pi)
  p1 = [np.cos(x) + 1j*np.sin(x) for x in a1]
  zs1 = PhaseConsistencyTest(p1, N)
print(zs1)
PolarHist(a1, 'polar_hist_0.01', title='Example with Phase Consistency 0.01')
N = 1600
sigma = (np.pi/2)*1.26
zs2 = 0
while zs2 < 0.0199 or zs2 > 0.0201:
  a2 = np.random.normal(np.pi, sigma, N)%(2*np.pi)
  p2 = [np.cos(x) + 1j*np.sin(x) for x in a2]
  zs2 = PhaseConsistencyTest(p2, N)
print(zs2)
PolarHist(a2, 'polar_hist_0.02', title='Example with Phase Consistency 0.02')
N = 1600
sigma = (np.pi/2)*1.191
zs3 = 0
while zs3 < 0.0299 or zs3 > 0.0301:
  a3 = np.random.normal(np.pi, sigma, N)%(2*np.pi)
  p3 = [np.cos(x) + 1j*np.sin(x) for x in a3]
  zs3 = PhaseConsistencyTest(p3, N)
print(zs3)
PolarHist(a3, 'polar_hist_0.03', title='Example with Phase Consistency 0.03')
N = 1600
sigma = (np.pi/2)*1.114
zs4 = 0
while zs4 < 0.0399 or zs4 > 0.0401:
  a4 = np.random.normal(np.pi, sigma, N)%(2*np.pi)
  p4 = [np.cos(x) + 1j*np.sin(x) for x in a4]
  zs4 = PhaseConsistencyTest(p4, N)
print(zs4)
PolarHist(a4, 'polar_hist_0.04', title='Example with Phase Consistency 0.04')
import string
# Lock down one random generation for plot:
num_bins = 128
bins = np.linspace(0, 2*np.pi, num_bins+1)[:-1] + 2*np.pi/(2*num_bins)
h1 = [11, 12, 10, 7, 7, 9, 10, 14, 6, 10, 5, 18, 10, 10, 10, 11, 15, 14, 7, 4, 7, 7, 12, 10, 16, 17, 12, 12, 7, 10, 13, 20, 13, 19, 6, 10, 11, 18, 9, 11, 22, 22, 15, 10, 16, 15, 17, 18, 9, 18, 18, 28, 21, 18, 17, 13, 14, 13, 17, 8, 11, 13, 9, 12, 21, 7, 16, 18, 13, 18, 20, 13, 10, 12, 12, 15, 13, 9, 15, 8, 19, 12, 12, 8, 17, 12, 12, 21, 9, 13, 14, 13, 10, 15, 14, 13, 10, 13, 18, 10, 16, 9, 7, 12, 11, 12, 9, 10, 10, 13, 15, 13, 10, 15, 13, 8, 5, 14, 3, 13, 9, 12, 8, 11, 11, 5, 17, 10]
h2 = [8, 11, 11, 12, 12, 9, 10, 9, 9, 8, 7, 9, 9, 7, 10, 15, 5, 11, 9, 10, 7, 15, 7, 10, 3, 5, 9, 10, 7, 14, 12, 11, 9, 9, 14, 15, 13, 10, 20, 14, 19, 15, 10, 8, 18, 17, 13, 15, 15, 12, 18, 17, 11, 16, 17, 14, 15, 18, 20, 21, 13, 18, 20, 20, 13, 22, 21, 10, 13, 13, 15, 10, 16, 17, 18, 13, 20, 13, 17, 13, 16, 14, 13, 18, 10, 18, 11, 11, 14, 14, 15, 11, 15, 12, 8, 21, 12, 10, 8, 6, 11, 18, 19, 10, 15, 10, 14, 14, 7, 12, 11, 16, 11, 18, 12, 12, 12, 11, 10, 6, 13, 15, 6, 6, 3, 14, 2, 6]
h3 = [4, 7, 5, 12, 4, 8, 9, 8, 10, 5, 9, 9, 3, 7, 11, 9, 10, 17, 11, 12, 9, 13, 11, 6, 10, 10, 13, 10, 12, 15, 12, 20, 8, 8, 11, 14, 14, 18, 16, 6, 17, 19, 14, 13, 15, 16, 27, 13, 16, 14, 17, 16, 19, 14, 11, 19, 23, 24, 21, 16, 12, 22, 11, 22, 16, 16, 15, 15, 18, 19, 15, 17, 15, 14, 10, 15, 12, 16, 11, 17, 17, 10, 14, 17, 16, 15, 13, 15, 16, 12, 11, 16, 12, 10, 16, 7, 8, 14, 20, 8, 17, 16, 7, 7, 9, 9, 10, 14, 10, 14, 10, 5, 4, 11, 14, 12, 6, 7, 13, 14, 8, 8, 8, 7, 10, 9, 8, 8]
h4 = [12, 6, 7, 8, 9, 6, 8, 7, 6, 8, 7, 8, 8, 5, 6, 1, 11, 12, 8, 9, 5, 5, 14, 8, 10, 10, 10, 14, 9, 11, 11, 15, 12, 19, 21, 14, 14, 15, 13, 19, 11, 14, 19, 17, 20, 14, 15, 16, 13, 21, 24, 18, 16, 14, 26, 16, 17, 14, 18, 22, 16, 13, 14, 21, 17, 19, 16, 19, 14, 22, 14, 23, 10, 19, 13, 12, 15, 19, 18, 15, 14, 5, 19, 9, 15, 14, 17, 17, 18, 13, 17, 13, 9, 10, 16, 8, 15, 4, 11, 10, 10, 10, 12, 12, 10, 16, 10, 7, 11, 10, 16, 9, 10, 9, 6, 7, 10, 12, 8, 14, 9, 10, 6, 5, 9, 5, 7, 11]
StartFig(figsize=(12,4))
plottups = [(h1, '0.01'), (h2, '0.02'), (h4, '0.04')]
for i,(h, label) in enumerate(plottups):
  ax = plt.subplot(1, len(plottups), i+1, projection='polar')
  bars = ax.bar(bins, h, width=2*np.pi/num_bins)
  plt.title(f'Phase Consistency {label}')
  ax.text(-0.2, 1.05, f'{string.ascii_lowercase[i]}.', transform=ax.transAxes, size=20, weight='bold')
  for b in bars:
    b.set_alpha(0.7)
plt.subplots_adjust(wspace=0.3)
SaveFig('polar_hists_range')
print('0.02 has', sum(e.real<0 for e in p2)/len(p2), 'to the left')
print('0.03 has', sum(e.real<0 for e in p3)/len(p3), 'to the left')
print('0.04 has', sum(e.real<0 for e in p4)/len(p4), 'to the left')
StartFig()
h, bins = plt.histogram(zs_arr)
plt.plot((bins[:len(bins)-1]+bins[1:])/2, h)
plt.plot([np.mean(zs_arr)]*2, [0, max(h)])
SaveFig('normality_test_zs')
print('min', min(zs_arr), 'max', max(zs_arr))
print(h)
print(bins)
print(np.mean(zs_arr), np.sum(np.array(h)*(bins[:len(bins)-1]+bins[1:])/2)/np.sum(h))
sigmult = [0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1, 1.125, 1.25, 1.5, 2]
zsv = [0.9976, 0.9904, 0.962, 0.857, 0.7067, 0.5398, 0.3817, 0.250, 0.0850, 0.0437, 0.02113, 0.00379, 0.00106]
StartFig()
plt.plot(zsv, sigmult)
plt.xlabel('zs value')
plt.ylabel('Sigma multiple of pi/2')
SaveFig('zs_exploration')
def zs_to_sigma_mult(zs):
  if zs <= 0:
    return 0
  # Estimate
  return np.arccos(zs**0.5)/(np.pi/2)-np.log(zs)/(2*np.pi)
for zs,sigm in zip(zsv, sigmult):
  print(zs_to_sigma_mult(zs), sigm)

