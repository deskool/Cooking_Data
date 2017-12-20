function [] = plot_NN_coefficients(train_AUC,test_AUC,weights)
% Plot the results
plot(train_AUC, test_AUC, '.')
xlabel('Training Performance')
ylabel('Testing Performance')

ind = 1;
for i = 1:10
    for j = 1:5
        
        % Add Labels
        if ind == 46
            xlabel('Fold Number')
        end
        if ind == 22
            ylabel('Weights')
        end
        
        subplot(5,10,ind);
        scatter(1:100,squeeze(weights(i,j,:)),4,test_AUC,'.');
        ind = ind+1;
           
    end
end
end

