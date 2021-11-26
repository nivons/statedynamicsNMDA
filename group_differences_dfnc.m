% the following code is based on the output from the "Temporal  dynamic  FNC
% toolbox  (dFNC)" that is part of GroupICATv4.0b (GIFT): 
% prefix_dfnc_cluster_stats.mat: dfnc_corrs (participant*connectivityValues*state)

% group comparison performed with: 
% bramila_ttest2_np.m from https://version.aalto.fi/gitlab/BML/bramila



% -----------------------------------
% add necessary functions & load data
% -----------------------------------

addpath('path\to\bramila-master\');
addpath(genpath('path\to\GroupICATv4.0b'));


load('path\to\GIFToutput\prefix_dfnc.mat');
load('path\to\GIFToutput\prefix_dfnc_cluster_stats.mat', 'dfnc_corrs');
mat = squeeze(dfnc_corrs);


% ------------------------------------
% group comparison of states
% ------------------------------------

% save state vectors in a struct

for f = 1:size(mat,3)
    
    s.(['state' num2str(f)]) = squeeze(mat(:,:,f));
    
end


% specify parameters for permutation testing

N_patients          = 61;
N_healthy_controls  = 57;

design              = [ones(1,N_patients) 2*ones(1,N_healthy_controls)];
niter               = 10000; % number of permutations
g1                  = find(design==1);
g2                  = find(design==2);


% perform group comparison and save into struct

rng('default')                                                          % save seed: https://de.mathworks.com/help/matlab/ref/rng.html
fields = fieldnames(s);

for i = 1:length(fields) % loop over states
        
    data                    = s.(fields{i});                            % extract connectivity values for state i
    data                    = permute(data, [2,1]);                     % rearrange matrix to value*partcipant
    [stats.(fields{i})]     = bramila_ttest2_np(data, design, niter);   % run permutation testing
                                                          
end

% tvals = t-values for datapoint, positive tvals mean group 1 > group 2
% pvals = the first column of the p value returns the p-values for group1 > group2
% the second column group1 < group2



% ----------------------------------
% perform FDR correction & save results
% ----------------------------------

p_value = .05;

for j = 1:length(fields) % loop over states
    
    % perform FDR correction
    stats.(fields{j}).pvals_corr         = mafdr(stats.(fields{j}).pvals(:),'BHFDR','true');
    stats.(fields{j}).pvals_corr_HC_NMDA = stats.(fields{j}).pvals_corr(1:size(data,1));
    stats.(fields{j}).pvals_corr_NMDA_HC = stats.(fields{j}).pvals_corr(size(data,1)+1:end);
    
    
    % find indices of the comparisons where we can reject the null hypothesis
    stats.(fields{j}).idx_HC_NMDA       = find(stats.(fields{j}).pvals_corr_HC_NMDA < p_value);
    stats.(fields{j}).idx_NMDA_HC       = find(stats.(fields{j}).pvals_corr_NMDA_HC < p_value); 
    
    
    % find significant connections in matrix for HC > NMDA
    a                                   = stats.(fields{j}).pvals_corr_HC_NMDA;
    a(a>p_value)                        = 0;
    stats.(fields{j}).mat_HC_NMDA       = icatb_vec2mat(a); % convert into matrix
    
    
    % find significant connections in matrix for NMDA > HC
    b                                   = stats.(fields{j}).pvals_corr_NMDA_HC;
    b(b>p_value)                        = 0;
    stats.(fields{j}).mat_NMDA_HC       = icatb_vec2mat(b); 
    
    
    % get component vector 
    stats.(fields{j}).components        = dfncInfo.comps;
end



% save struct with results
save('results.mat', 'stats');