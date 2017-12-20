function [] = plot_coefficients(t,number_of_random_seeds,odds_ratios,test_AUC)
feature_labels = t.Properties.VariableNames(1:end-1);                                   % get the names of the features
num_features = length(feature_labels);                                                  % get the number of features
for i = 1:num_features                                                                  % for each of the features...
    subplot(floor(num_features/10)+1,mod(num_features,10),i);                           % set up a subplot grid
                                                            
    
    scatter(1:number_of_random_seeds,odds_ratios(i+1,:),num_features,test_AUC)          % plot the odds ratio for each fold, and color by AUC on test  
    title(feature_labels{i})                                                            % title each subplot with the feature name
    hold on;                                                                            % keep working with this plot...
    
    if i == 1                                                                           % plot the y-label
        ylabel('Odds Ratio (e^{\beta})')
    end   
    if i == 3                                                                           % plot the x-label
        xlabel('Random Fold Number')
    end
    
    plot([0 number_of_random_seeds],[1 1],'black--')                                    % plot a black dashed line along y=1 (signifies no effect)
    xlim([0 number_of_random_seeds])                                                    % set the x-lim
    ylim([0 3])                                                                         % set the y-lim
    
end
colormap 'jet';                                                                         % color the points (hotter = higher test set AUC)
colorbar off

end