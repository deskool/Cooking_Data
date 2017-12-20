function [train_AUC,test_AUC] = use_nn_random_weights_loocv(t)
clear train_AUC test_AUC
for j = 1:10
    rng(j);
    for i = 1:height(t)
        tt = t;
        %Generate a random 70% Training, 30% Testing Split
        testing = tt(i,:); tt(i,:) = [];
        training = tt;
        
        tr_inputs = training{:,1:end-1}';                         % convert the traing set into a format the NN can work with
        tr_targets = training.Outcome';   
        te_inputs = testing{:,1:end-1}';                          % convert the testing set into a format the NN can work with
        te_targets = testing.Outcome';

        
        %Create Two-Layer Feedforward Network
        net = patternnet([10,10]);
        net.trainParam.showWindow = false;
        net.divideParam.trainRatio = .7;
        net.divideParam.valRatio =.3;
        net.divideParam.testRatio = 0;
        net = train(net,tr_inputs,tr_targets);
          
        train_predictions = net(tr_inputs);
        test_predictions(i) = net(te_inputs);
        
        % Evaluate the AUROC
        [~,~,~,train_AUC(i,j),~] = perfcurve(training.Outcome,train_predictions,1);
        
    end
    % histogram(train_AUC)
    % mean(train_AUC)
    % std(train_AUC)
    [~,~,~,test_AUC(j),~] = perfcurve(t.Outcome,test_predictions,1);
    display([num2str(j) '/' 10])
end
end

