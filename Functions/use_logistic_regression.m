function [train_AUC,test_AUC,odds_ratios] = use_logistic_regression(t,formula)
clear train_AUC test_AUC                                           % clear out any prior results
% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_set_perc = 0.7;                                           % how much data will be used for training
testing_set_perc = 0.3;                                            % how much data will be used for testing
number_of_random_seeds = 100;                                      % decrease to speed things up, increase to test more rigously
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

index_ones = find(t.Outcome == 1);                                 % the indicies of 'Smokers'
index_zeros = find(t.Outcome == 0);                                % the indicies of 'Non-Smokers'

for seed = 1:number_of_random_seeds                                % for each of the random seeds
    rng(seed);                                                     % set the random number generator
    
    random_ones_index = randperm(length(index_ones));              % random index for 'Smokers'
    random_zeros_index = randperm(length(index_zeros));            % random index for 'Non-Smokers'
  
    trind_ones  = floor(length(index_ones)*training_set_perc);     % cut-off for training set 'Smokers'
    trind_zeors = floor(length(index_zeros)*training_set_perc);    % cut-off for training set 'Non-Smokers'
    
    training = [t(index_ones(random_ones_index(1:trind_ones)),:) ;...
                t(index_zeros(random_zeros_index(1:trind_zeors)),:)];
    
    testing = [t(index_ones(random_ones_index(trind_ones+1:end)),:) ;...
               t(index_zeros(random_zeros_index(trind_zeors+1:end)),:)];
    
    model = fitglm(training,formula,'distr','binomial');           % train the logistic regression model using the training set
    odds_ratios(:,seed) = exp(model.Coefficients.Estimate);        % save the odds ratios of the features
    train_predictions = predict(model,training);                   % predict the outcomes in the training set 
    test_predictions = predict(model,testing);                     % predict the outcomes in the testing set
    [~,~,~,train_AUC(seed),~] = perfcurve(training.Outcome,...     % evaluate model performance on the training set
                                          train_predictions,1);
    [~,~,~,test_AUC(seed),~] = perfcurve(testing.Outcome,...       % evaluate model performance on the testing set
                                         test_predictions,1);
end

end

