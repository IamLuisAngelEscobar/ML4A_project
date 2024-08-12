%% NRMSE Calculating Function
function rmse_value = rmse(y_train, y_pred)
    % Calculate the root mean squared error (RMSE)
    rmse_value = sqrt(mean((y_train - y_pred).^2));
    
end