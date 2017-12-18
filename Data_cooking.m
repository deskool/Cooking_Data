%% TITLE:   HOW TO COOK YOUR DATA (WITHOUT EVEN KNOWING IT) %%%%%%%%%%%%%%%
%  AUTHOR:  MOHAMMAD M. GHASSEMI, PHD CANDIDATE, MIT 
%  DATE:    DECEMBER 18TH, 2017

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SUMMARY: * BEWARE OF MODELS THAT ARE VALIDATED USING ONLY ONE FOLD: 
%             E.G. 70% TRAINING, 15% VALIDATION, 15% TESTING. 
%
%           * ALWAYS CROSS VALIDATE, AND ALWAYS PROVIDE THE VARIANCE
%             OF THE MODELS ACROSS THE FOLDS.
%
%           * FOR LEAVE-ONE-OUT CROSS VALIDATION, REPORT THE VARIANCE IN
%             PERFORMANCE ON THE TRAINING SETS.
%
%           * MOST OF ALL, JUST BE HONEST. SCIENCE IS ABOUT THE TRUTH, NOT
%             FLASHY JOURNALS, MEDIA, AND FAME.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
%% STEP 1: IMPORT THE SMOKING DATASET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load a simple dataset, and format it for predicting smokers
% Inputs:  Blood pressure (diastolic, systolic), Sex, Age and Weight
% Outputs: Smoker/Non-Smoker (Binary outcome)
load smoking.mat; t = dataset2table(smoking); t.Outcome = t.Smoker > 0;

%% STEP 2: TRAIN A LOGISTIC REGRESSION MODEL (ALL DATA) %%%%%%%%%%%%%%%%%%%
% Train a model that predicts the probability of smoker, given covariates:
formula = 'Outcome ~ 1 + DiastolicBP + Sex + Age + Weight + SystolicBP';
model = fitglm(t,formula,'distr','binomial');

% Predict the probability according to the model
predictions = predict(model,t);

% Compare these predictions to the truth using a popular metric:
% Area Under Reciever Operator Curve (AUROC)
[X,Y,T,AUC,OPTROCPT] = perfcurve(t.Outcome,predictions,1);

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. This yields an AUC of 0.93, that's great performance!
%    But of course, any skeptical machine learning user
%    would know not to trust this performance level because we used all the
%    data. So, there is a good possibility that we overfit!

%% STEP 3[A]: TRAIN A LOGISTIC REGRESSION WITH ONE TRAINING/TESTING FOLD %%
% This is a standard operating prcedure in many papers. We will partition 
% our data into training and testing sets. We will identify model parameters
% using the training data (70%), and then evaluate the performance of the model
% using the held out test set (30%).

% Anyone that has worked on machine learning problems before knows that the
% preise properties of the training and testing data sets can have dramatic
% implications for the kind of performance your model will report.
% To illustrate this, let's split the data 1000 times, using different
% random seeds, each time.
clear train_AUC test_AUC
for seed = 1:1000
    %Generate a random 70% Training, 30% Testing Split
    rng(seed); random_index = randperm(height(t));
    training = t(random_index(1:70),:);
    testing = t(random_index(71:100),:);
    
    %Train the model
    model = fitglm(training,formula,'distr','binomial');
    
    %Evaluate performance of the training and testing sets.
    train_predictions = predict(model,training);
    test_predictions = predict(model,testing);
    
    %Evaluate the AUROXCC
    [~,~,~,train_AUC(seed),~] = perfcurve(training.Outcome,train_predictions,1);
    [~,~,~,test_AUC(seed),~] = perfcurve(testing.Outcome,test_predictions,1);
end

% Let's see how sensitive the performance of the model is to the random seed
% we used to generate the training and testing sets. To do this, we will
% simply plot the training set AUROC, and testing set AUROC.
figure; subplot(1,2,1);plot(train_AUC,test_AUC,'.');
xlabel('Training Set Performance (AUROC)');ylabel('Testing Set Performance (AUROC)');

% We can also plot a distribution that compares the difference in the
% training and testing set performances, across the 1000 random splits that
% we trained the logistic regression model on
subplot(1,2,2);histogram(test_AUC-train_AUC,'BinWidth',0.01);
xlabel('(Test - Train) AUC'); ylabel('Number of Splits')

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% There are two things we notice about the results

% 1. As a function of the way we select the training and test split, we
%    observe huge fluctations in the performance of the model. On 
%    training sets, AUC fluctuates between 0.88 to 1.0. On testing sets
%    AUC fluctuates between 0.66 -- 1.0.

% 2. There is a negative association between training and test set
%    performance. Choosing a 70-30 split that provides high testing set 
%    performance, tends to come at the cost of the algorithm's performance 
%    on the training set (this is dispite the fact that the algorithm is
%    always trained on the training set).

% CONCLUSIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Never forget that a machine learning model is supposed to generalize what
%    was observed in the training set to the testing set. Hence, there should
%    not be large differences in model performance between the two sets. 
%    One of the tell-tale signs that there is something wrong with the model
%    are large deviations between the training and the testing set performance. 
%    Hence, it's important that you report both your training set performance.

% 2. Beware of models evaluated using a single fold! Performance levels can
%    exceed even those of a model trained using ALL the data.


%% STEP 3[B]: MODELS WITHOUT DIVERGING TRAIN-TEST PERFORMANCE
%  Even if the training and testing set performances are simmilar,
%  it provides no gaurentee that the the results are robust. They
%  may simply be the result of a luckier training-testing split.

%  In the previous scatter-plot, we saw that several of training and testing
%  performances were the same. Let's visualize the cases where they were:

same_performance_index = find(round((test_AUC-train_AUC)*100) == 0);
figure; histogram(train_AUC(same_performance_index),'BinWidth',0.01);
xlabel('Training and Testing AUC'); ylabel('Number of Splits')


% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. What's interesting here is that the training and testing
%    performance are THE SAME, but simply choosing the training and test set
%    strategically still allows us to obtain an AUC anywhere between 0.91 
%    and 0.95. That gain in performance is not trivial. Many papers are 
%    published on the basis of simmilar magnitudes of improved performance.

% 2. This means that its possible to choose a 70-30 split, such that the 
%    method appears to performs equally well on the training and testing 
%    set, and even outperforms a model trained on the ENTIRE dataset 
%    (0.95 versus 0.93)! Incredible!

%% STEP 3[C]: EFFECTS ON MODELS COEFFICIENTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  If we are using a linear model, the coefficients of the model are
%  interpretable. In the case of logistic regression, we can compute the
%  odds-ratio exp(coefficients) and can visualize how the odds-ratios
%  change across the many folds.

for i = 1:1000
    rng(i);
    random_index = randperm(height(t));
    training = t(random_index(1:70),:);
    testing = t(random_index(71:100),:);
    
    %Train the model
    model = fitglm(training,formula,'distr','binomial');
    odds_ratios(:,i) = exp(model.Coefficients.Estimate);
end

% Plot the coefficients
feature_labels = {'Sex','Age','Weight','SystolicBP','DiastolicBP'};
for i = 1:5
    subplot(1,5,i);
    scatter(1:1000,odds_ratios(i+1,:),5,test_AUC)
    
    ylim([0 3])
    title(feature_labels{i})
    hold on;
    
    if i == 1
        ylabel('Odds Ratio')
    end
    
    if i == 3
        xlabel('Random Number Seed')
    end
    
    hold on;
    plot([0 1000],[1 1],'black--')
    
end
colormap 'jet'; h = colorbar;
ylabel(h, 'AUC of Model')

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 1. We see here that the composition of our training/test sets models can
%    have drastic implications for how we interpret some of our model
%    coefficients. The sex of the individual, for example, can range from 
%    having no predictive effect, to being strongly associated with Males (0), 
%    or Females (> 1), all depending on how the data was sliced.

%% STEP 3[D]: EFFECTS OF BALANCING OUTCOME DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Could it be possible that the large variance in the results are because
%  we didn't balance the outcomes in the training and testing sets? Let's
%  balance the data, and do the analysis again.

index_ones = find(t.Outcome == 1);
index_zeros = find(t.Outcome == 0);
clear train_AUC test_AUC
for seed = 1:1000
    %Generate a random 70% Training, 30% Testing Split
    rng(seed);
    random_ones_index = randperm(length(index_ones));
    random_zeros_index = randperm(length(index_zeros));
    
    trind_ones  = floor(length(index_ones)*0.7);
    trind_zeors = floor(length(index_zeros)*0.7);
    
    training = [t(index_ones(random_ones_index(1:trind_ones)),:) ;...
        t(index_zeros(random_zeros_index(1:trind_zeors)),:)];
    
    testing = [t(index_ones(random_ones_index(trind_ones:end)),:) ;...
        t(index_zeros(random_zeros_index(trind_zeors:end)),:)];
      
    %Train the model
    model = fitglm(training,formula,'distr','binomial');
    
    %Evaluate performance of the training and testing sets.
    train_predictions = predict(model,training);
    test_predictions = predict(model,testing);
    
    %Evaluate the AUROXCC
    [~,~,~,train_AUC(seed),~] = perfcurve(training.Outcome,train_predictions,1);
    [~,~,~,test_AUC(seed),~] = perfcurve(testing.Outcome,test_predictions,1);
end

%Notice what happens when we plot the training versus testing set performances
figure;
subplot(1,2,1);plot(train_AUC,test_AUC,'.'); %training
xlabel('Training Set Performance');ylabel('Testing Set Performance');

subplot(1,2,2);histogram(test_AUC-train_AUC,'BinWidth',0.01); % testing
xlabel('(Test - Train) AUC'); ylabel('Counts')

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. So, as a function of the random seed, even with an outcome balanced 
%    training and testing sets, we can still manipute the AUC values on the
%    test set to range from 0.70, to 1.0 (perfect), and 0.89 to 1.0 on the
%    training set.

%% STEP 4[A]: TRAIN A NEURAL NETWORK MODEL WITH ONE TRAINING/TESTING SPLIT.
%  Everything we've seen so far has assumed we are using a logistic
%  regression model, but what if we decide to use a more advanced method?
%  These days, neural netwoks are all the rage. So, let's try them next.

%  As before, let's split the data (70% training, 30% testing), and see
%  how sensitive a 10x10 feedforward neural network is to the way that the
%  data is split. As a pre-caution, we'll balance the outcomes in the
%  training and testing sets.


index_ones = find(t.Outcome == 1);
index_zeros = find(t.Outcome == 0);
clear train_AUC test_AUC
for seed = 1:100
    %Generate a random 70% Training, 30% Testing Split
    rng(seed);
    random_ones_index = randperm(length(index_ones));
    random_zeros_index = randperm(length(index_zeros));
    
    trind_ones  = floor(length(index_ones)*0.7);
    trind_zeors = floor(length(index_zeros)*0.7);
    
    training = [t(index_ones(random_ones_index(1:trind_ones)),:) ;...
        t(index_zeros(random_zeros_index(1:trind_zeors)),:)];
    
    tr_inputs = [training.Age' ; training.Weight' ; training.DiastolicBP' ;...
        training.Sex' ; training.SystolicBP'];
    tr_targets = training.Outcome';
    
    testing = [t(index_ones(random_ones_index(trind_ones:end)),:) ;...
        t(index_zeros(random_zeros_index(trind_zeors:end)),:)];
    
    te_inputs = [testing.Age'; testing.Weight' ; testing.DiastolicBP' ;...
        testing.Sex' ; testing.SystolicBP'];
    te_targets = testing.Outcome';
    
    %Create Two-Layer Feedforward Network
    rng(1);
    net = patternnet([10,10]);
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = .7;
    net.divideParam.valRatio =.3;
    net.divideParam.testRatio = 0;
    net = train(net,tr_inputs,tr_targets);
    
    
    train_predictions = net(tr_inputs);
    test_predictions = net(te_inputs);
    
    
    % Evaluate the AUROXCC
    [~,~,~,train_AUC(seed),~] = perfcurve(training.Outcome,train_predictions,1);
    [~,~,~,test_AUC(seed),~] = perfcurve(testing.Outcome,test_predictions,1);
    
    seed
end

plot(train_AUC,test_AUC,'.')
xlabel('Training Performance')
ylabel('Testing Performance')

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Just as we observed for the logistic regression, there are huge
%    variances in the performance of the models as a function of the
%    training/ testing split. We can choose a training/test split that 
%    achieves up to 0.98 on both the training and the testing sets, 
%    even for the SAME initialization of the weights matrix.

%% STEP 4[B]: HOW THE INITIALIZATION OF WEIGHTS IMPACT PERFORMACE ON A GIVEN FOLD.
%  By now, I hope it is painfully obvious that results can be easilly
%  maniputed on single-fold studies by simply engineering the training and
%  testing sets. Of course, most scientific investigators are not so
%  nefarious that they would intentionally choose the training and testing
%  folds this way. What is more common is for investigators to choose a
%  given training and testing fold, and to then optimize their algorithm
%  around it.

%  The problem is that investigators may train multiple models on the
%  training set, and then (to determine which is best) choose the one with the
%  best peformance on the testing set.

%  This process, however, is problematic in the same way that engineering 
%  the training/testing split was. For any given split, we can also identify a 
%  random weight that creates bloated performance on the training/testing sets.


index_ones = find(t.Outcome == 1);
index_zeros = find(t.Outcome == 0);
clear train_AUC test_AUC

rng(1);
random_ones_index = randperm(length(index_ones));
random_zeros_index = randperm(length(index_zeros));

trind_ones  = floor(length(index_ones)*0.7);
trind_zeors = floor(length(index_zeros)*0.7);

%training
training = [t(index_ones(random_ones_index(1:trind_ones)),:) ;...
            t(index_zeros(random_zeros_index(1:trind_zeors)),:)];
tr_inputs = [training.Age' ; training.Weight' ; training.DiastolicBP' ;...
             training.Sex' ; training.SystolicBP'];
tr_targets = training.Outcome';

%testing
testing = [t(index_ones(random_ones_index(trind_ones:end)),:) ;...
           t(index_zeros(random_zeros_index(trind_zeors:end)),:)];
te_inputs = [testing.Age'; testing.Weight' ; testing.DiastolicBP' ;...
             testing.Sex' ; testing.SystolicBP'];
te_targets = testing.Outcome';

%  Create Two-Layer Feedforward Network
for seed = 1:100
    rng(seed);
    net = patternnet([10,10]);
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = .7;
    net.divideParam.valRatio =.3;
    net.divideParam.testRatio = 0;
    net = train(net,tr_inputs,tr_targets);
   
    train_predictions = net(tr_inputs);
    test_predictions = net(te_inputs);
       
    %Evaluate the AUROC
    [~,~,~,train_AUC(seed),~] = perfcurve(training.Outcome,train_predictions,1);
    [~,~,~,test_AUC(seed),~] = perfcurve(testing.Outcome,test_predictions,1);  
    weights(:,:,seed) = net.IW{1};
    
    seed
end

% Plot the results
plot(train_AUC, test_AUC, '.')
xlabel('Training Performance')
ylabel('Testing Performance')

ind = 1;
for i = 1:10
    for j = 1:5
        
        % Add Labels
        if ind == 46
            xlabel('Split Number')
        end
        if ind == 22
            ylabel('Weights')
        end
        
        subplot(5,10,ind);
        scatter(1:100,squeeze(weights(i,j,:)),4,test_AUC,'.');
        ind = ind+1;
           
    end
end

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Once again, we see large variance in the performance on the models as
%    a function of the initial weights. This is itself not the problem,
%    rather, the problem is that an investigator might be tempted to simply
%    choose the initialization that gave the best performance on the
%    training AND TESTING SETS. In this case, it would provide: 0.98 AUC on
%    the training set and 0.96 on the testing set. 
%
% 2. Unlike linear models, the weights of Neural Networks arn't
%    interpretable.

%% 4[C]: THE EFFECTS OF TOPOLOGY ON PERFORMANCE
%  So what if we choose a given fold, and givn initialization of the
%  network weights? Well, there is still the topology of the network that
%  the investigator can tweak and (once again) that can lead to 
%  over-engineering the results 

% OUTCOME BALANCED DATA
index_ones = find(t.Outcome == 1);
index_zeros = find(t.Outcome == 0);
clear train_AUC test_AUC

rng(1);
random_ones_index = randperm(length(index_ones));
random_zeros_index = randperm(length(index_zeros));

trind_ones  = floor(length(index_ones)*0.7);
trind_zeors = floor(length(index_zeros)*0.7);

training = [t(index_ones(random_ones_index(1:trind_ones)),:) ;...
    t(index_zeros(random_zeros_index(1:trind_zeors)),:)];

tr_inputs = [training.Age' ; training.Weight' ; training.DiastolicBP' ;...
    training.Sex' ; training.SystolicBP'];
tr_targets = training.Outcome';

testing = [t(index_ones(random_ones_index(trind_ones:end)),:) ;...
    t(index_zeros(random_zeros_index(trind_zeors:end)),:)];

te_inputs = [testing.Age'; testing.Weight' ; testing.DiastolicBP' ;...
    testing.Sex' ; testing.SystolicBP'];
te_targets = testing.Outcome';


clear train_AUC test_AUC
ind = 1
%Create Two-Layer Feedforward Network
for i = 1:10
    for j = 1:10
        rng(1);
        net = patternnet([i,j]);
        net.trainParam.showWindow = false;
        net.divideParam.trainRatio = .7;
        net.divideParam.valRatio =.3;
        net.divideParam.testRatio = 0;
        net = train(net,tr_inputs,tr_targets);
        
        train_predictions = net(tr_inputs);
        test_predictions = net(te_inputs);
        
        %Evaluate the AUROXCC
        [~,~,~,train_AUC(ind),~] = perfcurve(training.Outcome,train_predictions,1);
        [~,~,~,test_AUC(ind),~] = perfcurve(testing.Outcome,test_predictions,1);
        ind = ind+1
        
    end
end


% Plot the results
scatter(train_AUC, test_AUC)
xlabel('Training Performance')
ylabel('Testing Performance')

% RESULTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Here again, the variance in performance persits. We can get 0.98 testing 
%    and 0.97 training by topology optimization.

%% STEP 5[A]: PART OF THE SOLUTION: LEAVE-ONE-OUT CROSS VALIDATION (LOOCV)
%  The solution to this over-engineering is simple. Be rigourous, and use
%  leave-one-out cross validation. Here's an example of how to do that with
%  the logistic regression model.

clear train_AUC test_AUC
for i = 1:height(t)
    tt = t;
    testing = tt(i,:); tt(i,:) = [];
    training = tt;
    
    %Train the model
    model = fitglm(training,formula,'distr','binomial');
    
    %Evaluate performance of the training and testing sets.
    train_predictions = predict(model,training);
    test_predictions(i) = predict(model,testing);
    
    %Evaluate the AUROXCC
    [~,~,~,train_AUC(i),~] = perfcurve(training.Outcome,train_predictions,1);
end

[~,~,~,test_AUC,~] = perfcurve(t.Outcome,test_predictions,1);

mean(train_AUC)
std(train_AUC)
test_AUC
% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Here, our performance is worse, but it's also more believable 
%    because LOOCV does not allow us to get getting lucky with 
%    the training/testing fold split. The Test AUC was 0.90
%    The mean AUC of the training models is 0.93 +/- 0.003

%% STEP 5[A]: HACKING THE LOOCV USING NEURAL NETWORKS
%  But even with LOOCV, peaking at the testing set can still allows us to 
%  engineer our results. This is particularly relevant for Neural Networks
%  because they have so many hyper-paramters we can modify. Let's try LOOCV
%  after initializing the weights to 10 different values:

clear train_AUC test_AUC
for j = 1:10
    rng(j);
    for i = 1:height(t)
        tt = t;
        %Generate a random 70% Training, 30% Testing Split
        testing = tt(i,:); tt(i,:) = [];
        training = tt;
        
        tr_inputs = [training.Age' ; training.Weight' ; training.DiastolicBP';...
            training.Sex' ; training.SystolicBP'];
        tr_targets = training.Outcome';
        
        te_inputs = [testing.Age'; testing.Weight' ; testing.DiastolicBP' ;...
            testing.Sex' ; testing.SystolicBP'];
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
    j
end

test_AUC
xlabel('Training Performance')
ylabel('Testing Performance')

% RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Here we see that even the LOOCV testing set performance can be
%    manipulated. It ranges from 0.86 to 0.92!!! So... can we trust these 
%    results? what is to be done...?

%% CONCLUSIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Every time you look at the model's performance on the testing set, be
% aware that you are doing exactly what the code in this little script is
% doing. You may never train on the data in the test sets, but your knowledge
% of the data can leak into the way you engineer the models to produce an
% over-engineered result that doesn't work in the real world.


