function [] = plot_results(train_AUC,test_AUC,number_of_random_seeds)

figure; 
plot(train_AUC,test_AUC,'.');                                                                % plot the training vs. testing performance                                  
title(['Training vs. Testing AUC for ' num2str(number_of_random_seeds) ' random folds'])     % plot the title
xlabel('Training Set Performance (AUC)');                                                    % plot the x-axis label
ylabel('Testing Set Performance (AUC)');                                                     % plot the y-axis label

hold on;                                                                                     % add to the current plot
plot([min(train_AUC) min(train_AUC)],[0 1],'black--');                                       % a line denoting lower bound for training
plot([max(train_AUC) max(train_AUC)],[0 1],'black--');                                       % a line denoting upper bound for training
plot([0 1],[min(test_AUC) min(test_AUC)],'red--');                                           % a line denoting lower bound for testing
plot([0 1],[max(test_AUC) max(test_AUC)],'red--');                                           % a line denoting upper bound for testing

same_performance_index = find(abs(round((test_AUC-train_AUC)*100)) <= 1);                    % find folds with simmilar training/testing performance
scatter(train_AUC(same_performance_index),test_AUC(same_performance_index),100,'red');       % denote them with a red circle

bpi = find(max([train_AUC(same_performance_index) + test_AUC(same_performance_index)]) == ...% find the fold with the most flattering performance
     [train_AUC(same_performance_index) + test_AUC(same_performance_index)]);
best_performance_index = same_performance_index(bpi);                                        
scatter(train_AUC(best_performance_index),test_AUC(best_performance_index),100,'green');     % denote the most flattering point with a green circle
text(train_AUC(best_performance_index)+.01,...                                               % generate text reporting the most flattering AUC <train,test>
     test_AUC(best_performance_index)+.01,...
     ['<' num2str(round(train_AUC(best_performance_index)*100)/100) ',',...
     num2str(round(test_AUC(best_performance_index)*100)/100) '>'],...
     'Color','green');

legend({'Model performance',...                                                              % generate the legend
        'training - lower bound',...
        'training - upper bound',...
        'testing - lower bound',...
        'testing - upper bound',...
        'simmilar train/test performance',...
        'most flattering results'},...
        'Location','northwest')                                                              % place the legend in the NW corner
         
ylim([0.5,1])                                                                                % set the x-axis range   
xlim([0.5 1])  
end% set the y-axis range