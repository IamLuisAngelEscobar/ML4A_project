%% NRMSE Calculating Function
function nrmse_value = calculate_nrmse(y_train, y_pred)
    % Calculate the root mean squared error (RMSE)
    rmse = sqrt(mean((y_train - y_pred).^2));
    
    % Calculate the range of the actual data
    data_range = max(y_train) - min(y_train);
    
    % Calculate the NRMSE
    nrmse_value = rmse / data_range;
end