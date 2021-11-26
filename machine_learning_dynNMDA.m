% This scripts runs an automatic LOOCV Machine Learning alghorithm on dynamic FC states
% accustomed to State-dependent Signatures of Anti-N-Methyl-D-Aspartate
% Receptor Encephalitis manuscript

% input:
% struct with predictors_dynFC including
% corr_pred_mat: participants*connectivityValuesOfAllStatesConcatenated
% col_indices:   indices of states, e.g. [1,742,1483,2224]
% corr_string_all_states: e.g. ["state1-5-6"] for all predictors

% written by Stephan Krohn



% -----------------------------
% load data and add toolboxes
% -----------------------------

addpath(genpath('path\to\L1General'));         % https://www.cs.ubc.ca/~schmidtm/Software/L1General.html
addpath(genpath('path\to\libsvm-master'));     % http://www.csie.ntu.edu.tw/~cjlin/libsvm

N_patients         = 57;
N_healthy_controls = 61;

cd to\outputfolder\



% % % % % % % % %  define % % % % % % % 

% dynamic or static
run_on_FC          =   'dynamic'            ; % run prediction pipeline on static or dynamic FC
input_state        =   '1'                  ; % if run_on_FC = 'dynamic', need to specify which input state to run classification on; can be 'all', '1', ..., '4'

% include only certain networks in analysis
net_fs              = 'yes'              ; % can be yes or no

% define components of networks of interest
VIS     = [11 38 87 90];
DMN     = [13 14 24 33 36 40 59 61 84 85];
FPN     = [12 28 29 51 54 71 89 91];

nets    = [VIS DMN FPN];

% specify regularization strengths
% e.g. linspace(0, 14, 1000) from 0 to 14 in 1000 steps
lambda_reg_strengths     =   linspace(0, 14, 1000)    ; 
lambda_reg_strengths_log =   linspace(0, .15, 1000)   ;


% % % % % % % % %  end % % % % % % % 





%  - - - - load dynamic FC - - - -
var_load                =   'predictors_dynFC_corrVals.mat'             ; % predictors_dynFC

tmp                     =   load(var_load)                              ;

corr_pred_mat           =   tmp.predictors_dynFC.corr_pred_mat          ;
col_indices             =   tmp.predictors_dynFC.col_indices            ;
corr_strings_allstates  =   tmp.predictors_dynFC.corr_strings_allstates ;



% - - - - use networks for feature selection - - - - %
% sort nets ascending AND descending to find all possible combinations
% (e.g. 10-91 and 91-10)
nets1 = sort(nets, 'ascend');
nets2 = sort(nets, 'descend');

% all possible dual combinations of network components
allpairs_asc  = nchoosek(nets1,2);
allpairs_desc = nchoosek(nets2,2);
allpairs      = [allpairs_asc; allpairs_desc];  % concatenate

% make a char out of components pairs
for p = 1:length(allpairs)
    a                  = allpairs(p,:);
    allpairs_char(p,:) = [num2str(a(1)) '-' num2str(a(2))];
end

% find char in corr_state_allstates
allpairs_nets = zeros(length(corr_strings_allstates),1);  % create empty "logical"

for c = 1:length(allpairs_char)
    TF = contains(corr_strings_allstates, allpairs_char(c,:));  % search pairs in corr_strings_allstates
    
    allpairs_nets = allpairs_nets + TF; % add to empty logical; sum(allpairs_nets) = 4*lower triangle of compojent combinations 
    clear TF
end



switch net_fs
    
    case 'yes'
        
        % replace variables
        
        corr_pred_mat(:,allpairs_nets==0)        = [];
        corr_strings_allstates(allpairs_nets==0) = [];
        n_nets                                   = size(allpairs_asc,1);
        col_indices                              = [1, (n_nets+1), (n_nets*2+1), (n_nets*3+1)];
        
end



%  - - - - compare with static FC - - - -
static_FC = load('path\to\sFC_permutation_test_results.mat');


idx_CON     =   1:N_healthy_controls                                        ; % row index control subjects
idx_NMDA    =   N_healthy_controls+1:N_healthy_controls + N_patients        ; % row index patients

Y_labels    =   [zeros(N_healthy_controls,1); ones(N_patients,1)]           ; % Y labels to predict: 61 controls, 57 patients



switch run_on_FC
    
    case 'static'
        
        
        input_data              =   static_FC.sFC                                                      ; % use static FC data as input (i.e., subject by predictor matrix, here 118 x 741)
        corr_strings            =   erase(corr_strings_allstates(1:col_indices(2)-1), ['state1-'])     ;
        Y_labels(Y_labels==0)   =   -1                                                                 ; % label controls as -1

        
    case 'dynamic'
        
        
        % parse which dynFC states to consider (all, 1, 2, 3, or 4)
        if strcmp(input_state, 'all')
            
            input_data          =       corr_pred_mat               ;
            corr_strings        =       corr_strings_allstates      ;
            
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those subjects where all states are present over the resting state
            
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels           =        Y_labels(state_present_idx)                                     ;
            
           
        elseif strcmp(input_state, '1')
            
            input_data          =       corr_pred_mat(:, col_indices(1):col_indices(2)-1)               ;
            corr_strings        =       corr_strings_allstates(col_indices(1):col_indices(2)-1)         ;
            
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those subjects where state 1 is present over the resting state
            
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels           =       Y_labels(state_present_idx)                                     ;
            
            
        elseif strcmp(input_state, '2')
            
            input_data          =       corr_pred_mat(:, col_indices(2):col_indices(3)-1)               ;
            corr_strings        =       corr_strings_allstates(col_indices(2):col_indices(3)-1)         ;
            
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those subjects where state 2 is present over the resting state
            
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
            
        elseif strcmp(input_state, '3')
            
            input_data          =       corr_pred_mat(:, col_indices(3):col_indices(4)-1)               ;
            corr_strings        =       corr_strings_allstates(col_indices(3):col_indices(4)-1)         ;
            
            
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those subjects where state 3 is present over the resting state
            
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
            
        elseif strcmp(input_state, '4')
            
            input_data          =       corr_pred_mat(:, col_indices(4):end)                            ;
            corr_strings        =       corr_strings_allstates(col_indices(4):end)                      ;
            
            state_present_idx   =       ~any(isnan(input_data),2)                                       ; % index those subjects where state 4 is present over the resting state
            
            input_data          =       input_data(state_present_idx,:)                                 ;
            Y_labels            =       Y_labels(state_present_idx)                                     ;
            
        end
        
        Y_labels(Y_labels==0)   = -1                                                                    ; % label controls as -1
        
        
        
end


% - - - feed filtered input data to prediction pipeline - - -

all_subjects_data   =   input_data  ; % to ensure downstream compatibility
labels              =   Y_labels    ;

switch run_on_FC
    case 'static'
        
        fprintf(['Running classification pipeline on static FC input:\n', num2str(length(labels)),' subjects: ', num2str(length(labels) - sum(Y_labels==1)),' controls, ', num2str(sum(Y_labels==1)),' patients...\n'])
        
        
    case 'dynamic'
        fprintf(['Running classification pipeline on dynFC input state ', input_state, ':\n', num2str(length(labels)),' subjects: ', num2str(length(labels) - sum(Y_labels==1)),' controls, ', num2str(sum(Y_labels==1)),' patients...\n'])
        
end




all_subjects_data = zscore(all_subjects_data);                            % application of z-score on each feature, for normalization



% identification of the important features at each cycle
N               =   size(all_subjects_data,1);
accuracy_all    =   nan(1,N)                 ;
prediction_all  =   nan(1,N)                 ;

accuracy_lassoglm       =   nan(1,N)    ;
bestlambda_lassoglm     =   nan(1,N)    ;
prediction_lassoglm     =   nan(1,N)    ;

features_all        =   cell(1,N)       ;
features_all_lasso  =   cell(1,N)       ;

opt         = struct()  ;    
opt.maxIter = 20        ;     % doing a specific number of iterations of the feature selection
opt.verbose = 0         ;     % do not print outputxels


% 
nVars = size(all_subjects_data,2);

% preallocate accuracy output
lambda_accuracy             =   nan(length(lambda_reg_strengths),2)         ;
lambda_accuracy(:,1)        =   lambda_reg_strengths                        ;
lambda_features             =   cell(length(lambda_reg_strengths),3)        ; % first col: selected features, second col: accuracy, third col: predictions


lambda_accuracy_lasso             =   nan(length(lambda_reg_strengths),2)         ;
lambda_accuracy_lasso(:,1)        =   lambda_reg_strengths_log                        ;
lambda_features_lasso             =   cell(length(lambda_reg_strengths),3)        ; % first col: selected features, second col: accuracy, third col: predictions



for j = 1:length(lambda_reg_strengths)
    
    fprintf('\n\nHyperparameter iteration %d... \nFinished ', j)
    
    w_init = zeros(nVars+1,1);
    for i=1:N
        
        if mod(i,10)==0
            fprintf('%g...',i)
        end
        
        
        % defining a training set and a test set
        %     labels=labels(randperm(length(labels)));    % uncomment this and run the loop multiple time if you want to do permutation testing
        trainIndex = ones(N,1); trainIndex(i) = 0;      % all samples but the i'th are used for training
        testIndex = zeros(N,1); testIndex(i) = 1;       % only the i'th sample is used for testing
        trainData = all_subjects_data(trainIndex==1,:); trainLabel = labels(trainIndex==1,:);
        testData = all_subjects_data(testIndex==1,:); testLabel = labels(testIndex==1,:);
        
        nInstances = size(trainData,1);
        nVars = size(trainData,2);
        X = [ones(nInstances,1) trainData]; % Add Bias element to features
        y = sign(trainLabel); % Convert y to binary {-1,1} representation
        funObj = @(w)LogisticLoss(w,X,y);
        lambda = lambda_reg_strengths(j)*ones(nVars+1,1);
        lambda(1) = 0; 
        
        % running the classification
        wLogL1 = L1General2_PSSgb(funObj,w_init,lambda, opt);    % returns the weight matrix of the regression
        
        features_all{i}         =   find(wLogL1(2:end))                             ;  % the selected features are the non-zeroed elements of the weights matrix
        accuracy_all(i)         =   (sign(testData*wLogL1(2:end))==sign(testLabel)) ;  % the accuracy of the classification on the testing data
        prediction_all(i)       =   sign(testData*wLogL1(2:end))                    ;  % track prediction for subsequent confusion analysis
        
        
        

        % compare with inbuilt L1-regularized logistic regression function
        trainLabel(trainLabel==-1)  =   0                                                                                               ;
        [B_i, FitInfo_i]            =   lassoglm(trainData, trainLabel, 'binomial', 'Lambda', lambda_reg_strengths_log(j), 'Standardize', false, 'Alpha', 1)      ;
        
  
        coeffs_i                    =   [FitInfo_i.Intercept; B_i]                                          ;
        
        % which predictors were included with the given regularization
        % strength?
        features_all_lasso{1,i}     =   find(B_i)                                   ;
        
        % prediction
        yhat                        =   glmval(coeffs_i, testData, 'logit')                                                             ;
        yhatBinom                   =   (yhat>=0.5)                                                                                     ;
        
        % which class is predicted?
        if yhatBinom==1
            prediction_lassoglm(i)      =   1                                                                                           ;
        elseif yhatBinom==0
            prediction_lassoglm(i)      =   -1                                                                                          ;
        end
                                                                  
        
        % check accuracy of prediction
        accuracy_lassoglm(i)        =   prediction_lassoglm(i)==testLabel                                                               ;

        
    end
    
      
    
    lambda_accuracy_lasso(j,2)      =   mean(accuracy_lassoglm)     ;
    
    lambda_features_lasso{j,1}      =   features_all_lasso          ;
    lambda_features_lasso{j,2}      =   accuracy_lassoglm           ;
    lambda_features_lasso{j,3}      =   prediction_lassoglm         ;
    
    
    
    % SVM_overall_accuracy
    L1logreg_overall_accuracy = mean(accuracy_all)       ; % the average cross-validation prediction success, across cycles
    
    
    lambda_accuracy(j,2)    =   L1logreg_overall_accuracy       ;
    
    lambda_features{j,1}    =   features_all                    ;
    lambda_features{j,2}    =   accuracy_all                    ;
    lambda_features{j,3}    =   prediction_all                  ;
    
   
end



% find regularization strength with minimum CV-loss after grid search
[maxacc,maxacc_r]   =   max(lambda_accuracy(:,2))           ;
best_lambda           =   lambda_accuracy(maxacc_r,1)       ;



maxacc_feat         =   lambda_features(maxacc_r,:)     ;

maxacc_selected_feat    =   cell2mat(maxacc_feat{1,1}')         ;

feature_count           =   [1:nVars; histc(maxacc_selected_feat', 1:nVars)]'       ;
feature_count(:,3)      =   feature_count(:,2)/N                                    ;

feature_count_cell      =   num2cell(feature_count(feature_count(:,2)~=0,:))        ;

idx_selected_features   =   feature_count(feature_count(:,2)~=0,1)                  ;
feature_count_cell(:,4) =   cellstr(corr_strings(idx_selected_features))            ;

feature_count_cell      =   sortrows(feature_count_cell, [3], 'descend')            ;


% convert back to 0 and 1 to feed into confusion plot
idx_labels_neg_ones         =   labels==-1          ;
labels_conf                 =   labels              ;
labels_conf(idx_labels_neg_ones)=0                  ;

idx_preds_neg_ones          =   lambda_features{maxacc_r,3}'==-1          ;
labels_conf_pred            =   lambda_features{maxacc_r,3}'              ;
labels_conf_pred(idx_preds_neg_ones)=0                             ;



fh = figure;
plotconfusion(labels_conf', labels_conf_pred')

switch run_on_FC
    case 'static'        
title(['Static FC (n=', num2str(N), '; ', num2str(length(labels) - sum(Y_labels==1)),' CON, ', num2str(sum(Y_labels==1)),' NMDARE)'])
    case 'dynamic'
title(['dynFC state ', input_state, ' (n=', num2str(N), '; ', num2str(length(labels) - sum(Y_labels==1)),' CON, ', num2str(sum(Y_labels==1)),' NMDARE)'])
end


ah      = fh.Children(2)        ;
ah.XTickLabel{1}    = 'Control'  ;
ah.YTickLabel{1}    = 'Control'  ;
ah.XTickLabel{2}    = 'NMDARE'  ;
ah.YTickLabel{2}    = 'NMDARE'  ;
ah.XLabel.String    = 'True class'  ;
ah.YLabel.String    = 'Predicted'   ;

%fontsize
set(findobj(gca,'type','text'),'fontsize',13)

% defining my colors  
f1          = [0 0 139]/255;
nice_blue   = [67 147 195]/255;  % #4393C3
nice_red    = [214 96 77]/255;  % #D6604D
f14         = [85 26 139]/255;

% set(findobj(gca,'color',[0,102,0]./255),'color',nice_green)
set(findobj(gca,'color',[0,102,0]./255),'color',nice_blue)
set(findobj(gca,'color',[102,0,0]./255),'color',nice_red)
set(findobj(gcf,'facecolor',[120,230,180]./255),'facecolor',nice_blue)
set(findobj(gcf,'facecolor',[230,140,140]./255),'facecolor',nice_red)
set(findobj(gcf,'facecolor',[0.5,0.5,0.5]),'facecolor',[245, 245, 245]./255)
set(findobj(gcf,'facecolor',[120,150,230]./255),'facecolor','w')

saveas(ah, ['conf_state' input_state '.fig']);



% plot regularization optimization
figure
subplot(2,1,1)
plot(lambda_accuracy(:,1), lambda_accuracy(:,2))
xlabel('Regularization strength')
ylabel('Accuracy (LOOCV)')
ylim([0 1])

switch run_on_FC
    case 'static'        
        title('Regularization static FC')        
    case 'dynamic'
        title(['Regularization dynFC state ', input_state])
end

hold on
plot(lambda_accuracy(maxacc_r,1), lambda_accuracy(maxacc_r,2), 'r*')
legend('CV-Accuracy', ['Best Lambda = ', num2str(round(best_lambda,2))])
%hold off


freq_threshold              =       .1                                                    ; % lower threshold on how often a feature must have been selected over all LOOCVs
idx_featfreq_over_thresh    =       cell2mat(feature_count_cell(:,3))>=freq_threshold         ;


% figure
subplot(2,1,2)
bar(1:sum(idx_featfreq_over_thresh), cell2mat(feature_count_cell(idx_featfreq_over_thresh,3)), 'FaceColor', nice_blue)
ylim([0 1.1])

switch run_on_FC
    case 'static'        
title(['Feature Selection for static FC (threshold: >', num2str(freq_threshold),')'])
xlabs_split = corr_strings                  ;
    case 'dynamic'
title(['Feature Selection for dynFC state ', input_state, ' (threshold: >', num2str(freq_threshold),')'])
        xlabs_split = erase(feature_count_cell(:,4), ['state', input_state,'-'])                     ;
end
set(gca, 'TickLength', [0 0])
set(gca, 'XTick', 1:sum(idx_featfreq_over_thresh), 'XTickLabel', xlabs_split(idx_featfreq_over_thresh), 'XTickLabelRotation', 70)

xlabel('Features (Connections)')
ylabel('Selection Rate')

savefig(['reg_fs_state' input_state '.fig'])


% save results
save(['ML_state' input_state '_t' num2str(t_thresh) '.mat'], 'feature_count', 'feature_count_cell', ...
   'maxacc', 'input_state');




