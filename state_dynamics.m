% performs group comparison of dwell times, fraction times and transition
% frequencies

% dwell_times: contains mean dwell time in each state for all particpants 
% (Nparticipants*Nstate), NaN if not visited
% fraction_times: contains mean fraction time in each state for all particpants 
% (Nparticipants*Nstate), NaN if not visited
% trans_freq: contains absolute number of transitions between each pair of 
% states for all particpants 
% (Nparticipants*Npairofstates)

% group comparison performed with: 
% bramila_ttest2_np.m from https://version.aalto.fi/gitlab/BML/bramila



% -----------------------------------
% add necessary functions & load data
% -----------------------------------

addpath('path\to\bramila-master\');

load('path\to\workspace_dwelltimes_fractiontimes_transfreq.mat', ...
   'dwell_times', 'trans_freq');

N_patients          = 61;
N_healthy_controls  = 57;



% -----------------------------------
% group comparison on state dynamics
% -----------------------------------

rng('default')
niter               = 10000; % number of permutations
design              = [ones(1,N_healthy_controls) 2*ones(1,N_patients)];
g1                  = find(design==1);
g2                  = find(design==2);


% dwell_times
stats_dt            = bramila_ttest2_np(dwell_times', design, niter);

% fraction_times
stats_ft            = bramila_ttest2_np(fraction_times', design, niter);

%transition frequency
stats_tf            = bramila_ttest2_np(trans_freq', design, niter);


