% The following code implements the analyses presented in
% 
% "State-dependent Signatures of Anti-N-Methyl-D-Aspartate Receptor
% Encephalitis" (von Schwanenflug et al., 2021, Brain Communications),
% 
% including group comparisons of static and dynamic connectivity, state
% dynamics as well as classification analysis of group status (patients vs.
% controls) based on static and dynamic functional connectivity. 
% 
% 
% Dependencies:
% 
% Dynamic functional connectivity analyses are based on the output from 
% the "Temporal dynamic FNC toolbox (dFNC)" as part of the Group ICA of 
% fMRI Toolbox (GIFT), version: GroupICATv4.0b: 
% (https://trendscenter.org/software/gift).
% 
% Permutation test for group comparisons:
% (https://version.aalto.fi/gitlab/BML/bramila)
% 
% L1General for L1-regularized logistic regression: 
% (https://www.cs.ubc.ca/~schmidtm/Software/L1General.html)
% 
% Mathworks Statistics and Machine Learning Toolbox v12.1
% (https://de.mathworks.com/products/statistics.html)
%
% Mathworks Bioinformatics Toolbox v4.15.1
% (https://de.mathworks.com/products/bioinfo.html)
%
% Mathworks Parallel Computing Toolbox v7.4 & Server v7.4
% (https://de.mathworks.com/products/parallel-computing.html)
% 
% 
% Variables:
% Group comparison of states:
%   sampleData_dfnc_cluster_stats.mat: dfnc_corrs (Nparticipants*FC*state)
% 
% Group comparison of dwell times, fraction times and transition
% frequencies:
%   dwell_times: contains mean dwell time in each state for all particpants 
%       (Nparticipants*Nstate), NaN if not visited
%   fraction_times: contains mean fraction time in each state for all particpants 
%       (Nparticipants*Nstate), NaN if not visited
%   trans_freq: contains absolute number of transitions between each pair of 
%       states for all participants 
%       (Nparticipants*Npairofstates)
% 
% Input for classification on static and dynamic states is a structure with predictors_dynFC:
%   corr_pred_mat: participants*FCOfAllStatesConcatenated
%   corr_string_all_states: labels for predictor component connections
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
clearvars
clc

% -----------------------------------
% paths & data
% -----------------------------------

addpath('path\to\bramila-master\');
addpath('path\to\GroupICATv4.0b');
addpath('path\to\L1General');                               


load('componentLabels.mat');                                % load component labels
load('sampleData_dfnc_cluster_stats.mat', 'dfnc_corrs');    % load state vectors
load('sampleData_statedynamics.mat', 'dwell_times', 'fraction_times', 'trans_freq');    % load dynamic metrics


% specify parameters for permutation testing

N_patients          = 3;
N_healthy_controls  = 3;


design              = [ones(1,N_healthy_controls) 2*ones(1,N_patients)];
niter               = 10000;                                % number of permutations
g1                  = find(design==1);
g2                  = find(design==2);


% ------------------------------------
% group comparison of states
% ------------------------------------

mat = squeeze(dfnc_corrs);

% save state vectors in a struct

for f = 1:size(mat,3)
    
    s.(['state' num2str(f)]) = squeeze(mat(:,:,f));
    
end


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


% perform FDR correction & save results

p_value = .25;      % adjust accordingly

for j = 1:length(fields) % loop over states
    
    % perform FDR correction
    stats.(fields{j}).pvals_corr         = mafdr(stats.(fields{j}).pvals(:),'BHFDR','true'); % apply a rank-based multiple compariosn correction
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
    stats.(fields{j}).components        = comps;
end




% -----------------------------------
% group comparison on state dynamics
% -----------------------------------

% dwell_times
stats_dt            = bramila_ttest2_np(dwell_times', design, niter);

% fraction_times
stats_ft            = bramila_ttest2_np(fraction_times', design, niter);

%transition frequency
stats_tf            = bramila_ttest2_np(trans_freq', design, niter);



% -----------------------------------
% static & state-wise classification
% -----------------------------------

% define parameters for classification
run_on_FC          =   'dynamic'            ; % run prediction pipeline on static or dynamic FC
input_state        =   '2'                  ; % if run_on_FC = 'dynamic', need to specify which input state to run classification on; can be 'all', '1', ..., '4'

% define components of networks of interest according to Peer et al., 2017 (DOI: 10.1016/S2215-0366(17)30330-9)
VIS     = [11 38 87 90]                     ;
DMN     = [13 14 24 33 36 40 59 61 84 85]   ;
FPN     = [12 28 29 51 54 71 89 91]         ;
nets    = [VIS DMN FPN]                     ;

% specify grid search for regularization strength
lambda_reg_strengths     =   linspace(0, 14, 1000)    ; 

% load predictors for static or dynamic functional connectivity
switch run_on_FC 
    
    case 'static'
    var_load            = 'sampleData_predictors_sFC.mat'               ;
    
    case 'dynamic'

    var_load            = 'sampleData_predictors_dynFC.mat'             ;
end
tmp                     =   load(var_load)                              ;

corr_pred_mat           =   tmp.predictors_dynFC.corr_pred_mat          ;
corr_strings_allstates  =   tmp.predictors_dynFC.corr_strings_allstates ;

% network features, sort nets ascending AND descending to define component combinations
nets1 = sort(nets, 'ascend')    ;
nets2 = sort(nets, 'descend')   ;

% all possible combinations of network components
allpairs_asc  = nchoosek(nets1,2)               ;
allpairs_desc = nchoosek(nets2,2)               ;
allpairs      = [allpairs_asc; allpairs_desc]   ;  

% preallocate component pairs
allpairs_char = char(zeros(size(allpairs,1),5)); 

% components pairs
for p = 1:length(allpairs)    
    a                  = allpairs(p,:);
    allpairs_char(p,:) = [num2str(a(1)) '-' num2str(a(2))]  ;
end

% find char in corr_state_allstates
allpairs_nets = zeros(length(corr_strings_allstates),1)     ;  
for c = 1:length(allpairs_char)
    TF              = contains(corr_strings_allstates, allpairs_char(c,:))  ;   % search pairs in corr_strings_allstates
    allpairs_nets   = allpairs_nets + TF                                    ;   % add to empty logical; sum(allpairs_nets) = 4*lower triangle of component combinations  
end


% network features over states
corr_pred_mat(:,allpairs_nets==0)        = [];
corr_strings_allstates(allpairs_nets==0) = [];
n_nets                                   = size(allpairs_asc,1);
col_indices                              = [1, (n_nets+1), (n_nets*2+1), (n_nets*3+1)];


% full cohort
idx_CON     =   1:N_healthy_controls                                        ; % row index control participants
idx_NMDA    =   N_healthy_controls+1:N_healthy_controls + N_patients        ; % row index patients
Y_labels    =   [zeros(N_healthy_controls,1); ones(N_patients,1)]           ; % Y labels to predict: 61 controls, 57 patients

% get data
switch run_on_FC
    
    % static functional connectivity
    case 'static'
        
        input_data              =   static_FC.sFC                                                      ; % use static FC data as input
        corr_strings            =   erase(corr_strings_allstates(1:col_indices(2)-1), 'state1-')       ;
        Y_labels(Y_labels==0)   =   -1                                                                 ; % label controls as -1
        
    % dynamic functional connectivity    
    case 'dynamic'
        
        % parse which dynFC states to consider (all, 1, 2, 3, or 4)
        if strcmp(input_state, 'all')
            
            input_data          =       corr_pred_mat                   ;
            corr_strings        =       corr_strings_allstates          ;
            state_present_idx   =       ~any(isnan(input_data),2)       ; % index those participants where all states are present over the resting state            
            input_data          =       input_data(state_present_idx,:) ;
            Y_labels            =        Y_labels(state_present_idx)    ; 
           
        elseif strcmp(input_state, '1')
            
            input_data          =       corr_pred_mat(:, col_indices(1):col_indices(2)-1)               ;
            corr_strings        =       corr_strings_allstates(col_indices(1):col_indices(2)-1)         ;
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those participants where state 1 is present over the resting state            
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
        elseif strcmp(input_state, '2')
            
            input_data          =       corr_pred_mat(:, col_indices(2):col_indices(3)-1)               ;
            corr_strings        =       corr_strings_allstates(col_indices(2):col_indices(3)-1)         ;
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those participants where state 2 is present over the resting state
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
        elseif strcmp(input_state, '3')
            
            input_data          =       corr_pred_mat(:, col_indices(3):col_indices(4)-1)               ;
            corr_strings        =       corr_strings_allstates(col_indices(3):col_indices(4)-1)         ;
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those participants where state 3 is present over the resting state         
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
        elseif strcmp(input_state, '4')
            
            input_data          =       corr_pred_mat(:, col_indices(4):end)                            ;
            corr_strings        =       corr_strings_allstates(col_indices(4):end)                      ;
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those participants where state 4 is present over the resting state
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
        end
        
        % convert labels to 1 (patients) and -1 (controls) for downstream compatibility
        Y_labels(Y_labels==0)   =       -1                                                              ;
end

% prepare prediction
all_participants_data   =   input_data  ; % all available data for specified prediction (static vs. state-wise)
labels                  =   Y_labels    ;

% z-score features
all_participants_data   = zscore(all_participants_data)             ;                           

% number of predictor variables
nVars                   = size(all_participants_data,2)             ;

% preallocation 
N                       =   size(all_participants_data,1)           ;
iter_prediction         =   nan(1,N)                                ;
iter_accuracy           =   nan(1,N)                                ;
iter_features           =   cell(1,N)                               ;
lambda_accuracy         =   nan(length(lambda_reg_strengths),2)     ;
lambda_accuracy(:,1)    =   lambda_reg_strengths                    ;
lambda_features         =   cell(length(lambda_reg_strengths),3)    ; % first col: selected features, second col: accuracy, third col: predictions

% options structure for prediction
opt             = struct()  ;    
opt.maxIter     = 20        ;     % number of iterations for feature selection
opt.verbose     = 0         ;     % do not print verbose output

% feedback
switch run_on_FC
    case 'static'
        fprintf(['\nRunning classification pipeline on static FC input:\n', num2str(length(labels)),' participants: ', num2str(length(labels) - sum(Y_labels==1)),' controls, ', num2str(sum(Y_labels==1)),' patients...'])    
    case 'dynamic'
        fprintf(['\nRunning classification pipeline on dynFC input state ', input_state, ':\n', num2str(length(labels)),' participants: ', num2str(length(labels) - sum(Y_labels==1)),' controls, ', num2str(sum(Y_labels==1)),' patients...'])
end

% run prediction
for j = 1:length(lambda_reg_strengths)
        
    % initial weights
    w_init  =   zeros(nVars+1,1)    ;
    
    for i=1:N
        
        % training and test sets, prediction from L1-regularized logistic regression, follows Peer et al., 2017 (DOI: 10.1016/S2215-0366(17)30330-9)
        trainIndex      = ones(N,1)                             ; 
        trainIndex(i)   = 0                                     ; 
        testIndex       = zeros(N,1)                            ; 
        testIndex(i)    = 1                                     ;
        trainData       = all_participants_data(trainIndex==1,:); 
        trainLabel      = labels(trainIndex==1,:)               ;
        testData        = all_participants_data(testIndex==1,:) ; 
        testLabel       = labels(testIndex==1,:)                ;
        
        % prepare classification
        nInstances      = size(trainData,1)                         ;
        nVars           = size(trainData,2)                         ;
        X               = [ones(nInstances,1) trainData]            ;                                            % Add Bias element to features
        y               = sign(trainLabel)                          ;                                                          % Convert y to binary {-1,1} representation
        funObj          = @(w)LogisticLoss(w,X,y)                   ;
        lambda          = lambda_reg_strengths(j)*ones(nVars+1,1)   ;
        lambda(1)       = 0                                         ; 
        
        % run classification (calls L1General; https://www.cs.ubc.ca/~schmidtm/Software/L1General.html)
        wLogL1               =   L1General2_PSSgb(funObj,w_init,lambda, opt)     ;  % returns weights
        iter_features{i}     =   find(wLogL1(2:end))                             ;  % selected features are non-zeroed elements
        iter_accuracy(i)     =   (sign(testData*wLogL1(2:end))==sign(testLabel)) ;  % accuracy of prediction on test instance
        iter_prediction(i)   =   sign(testData*wLogL1(2:end))                    ;  % track prediction for subsequent confusion analysis

    end
       
    % track output
    L1_CV_accuracy          =   mean(iter_accuracy)      ; % average cross-validated prediction performance
    lambda_accuracy(j,2)    =   L1_CV_accuracy           ;
    lambda_features{j,1}    =   iter_features            ;
    lambda_features{j,2}    =   iter_accuracy            ;
    lambda_features{j,3}    =   iter_prediction          ;
    
end
fprintf('done.')




