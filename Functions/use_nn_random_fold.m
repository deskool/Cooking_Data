function [train_AUC,test_AUC] = use_nn_random_fold(t)
clear train_AUC test_AUC                                               % clear out any prior results
% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_set_perc = 0.7;                                               % how much data will be used for training
testing_set_perc = 0.3;                                                % how much data will be used for testing
number_of_random_seeds = 100;                                          % decrease to speed things up, increase to test more rigously
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

index_ones = find(t.Outcome == 1);                                     % the indicies of 'Smokers'
index_zeros = find(t.Outcome == 0);                                    % the indicies of 'Non-Smokers'

for seed = 1:number_of_random_seeds                                    % for each of the random seeds
    rng(seed);                                                         % set the random number generator    
    random_ones_index = randperm(length(index_ones));                  % random index for 'Smokers'
    random_zeros_index = randperm(length(index_zeros));                % random index for 'Non-Smokers'
    trind_ones  = floor(length(index_ones)*training_set_perc);         % cut-off for training set 'Smokers'
    trind_zeors = floor(length(index_zeros)*training_set_perc);        % cut-off for training set 'Non-Smokers'
    
    training = [t(index_ones(random_ones_index(1:trind_ones)),:) ;...  % generate an outcome balanced training set
                t(index_zeros(random_zeros_index(1:trind_zeors)),:)];  
    testing = [t(index_ones(random_ones_index(trind_ones+1:end)),:) ;...% generate an outcome blanaced testing set
               t(index_zeros(random_zeros_index(trind_zeors+1:end)),:)];
           
    tr_inputs = training{:,1:end-1}';                                   % convert the traing set into a format the NN can work with
    tr_targets = training.Outcome';   
    te_inputs = testing{:,1:end-1}';                                    % convert the testing set into a format the NN can work with
    te_targets = testing.Outcome';
  
    rng(1);                                                             % Require the initial weights of the NN to be the same, in each fold
    net = patternnet([10,10]);                                          % use a <inputx10x10x1> network 
    net.trainParam.showWindow = false;                                  % supress matlab popups
    net.divideParam.trainRatio = .7;                                    % internally, the network will use 70% of the training data for training
    net.divideParam.valRatio =.3;                                       % internally, the network will use 30% of the training data for validation 
    net.divideParam.testRatio = 0;                                      
    
    net = train(net,tr_inputs,tr_targets);                              % train the neural network on the training data
    train_predictions = net(tr_inputs);                                 % get the predictions on the training set
    test_predictions = net(te_inputs);                                  % get the predictions on the testing set
    
    [~,~,~,train_AUC(seed),~] = perfcurve(training.Outcome,...          % evaluate model performance on the training set
                                          train_predictions,1);
    [~,~,~,test_AUC(seed),~] = perfcurve(testing.Outcome,...            % evaluate model performance on the testing set
                                         test_predictions,1);
end
end

