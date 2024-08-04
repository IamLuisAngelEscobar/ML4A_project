%clear;
clc;
close all;


%% ENERGY PREDICTION

% Load Energy Data.

Train_Data = load("redhouseTrain.mat");

%Train_Data = Train_Data; 
d = Train_Data.d;
proxy = Train_Data.proxy;
t = Train_Data.t;
u = Train_Data.u;
x = Train_Data.x;
y = Train_Data.y;

%% STEP 1: Create Tables for energy

E_Col = {'time', 'Current Energy', 'R1','R2','R3','R4','R5', 'R6','R7','R8','R9','R10', 'Energy Outputs'};

E1 = [t', x(1,:)', x(4:13,:)',  y(1,:)'];
E1 = fillmissing(E1,"linear");

E2 = [t', x(2,:)', x(4:13,:)',  y(2,:)'];
E2 = fillmissing(E2,"linear");

E3 = [t', x(3,:)', x(4:13,:)',  y(3,:)'];
E3 = fillmissing(E3,"linear");

Tbl_E1 = array2table(E1, 'VariableNames',E_Col);
Tbl_E2 = array2table(E2, 'VariableNames',E_Col);
Tbl_E3 = array2table(E3, 'VariableNames',E_Col);

% store tables in Cell arrays. 
energy = {Tbl_E1,Tbl_E2,Tbl_E3};


%% STEP 2: Feature Selection For Lagging

for cellItem = 1:3
        
        Data = eval(['Tbl_E' num2str(cellItem)]);
        fs_model1 = fitlm(Data);
        coefficients = table2array(fs_model1.Coefficients(:, 'Estimate'));
        feature_names = Data.Properties.VariableNames;
        [~, idx] = sort(abs(coefficients(2:end)), 'descend');
        sorted_coefficients = coefficients(idx + 1);
        sorted_feature_names = feature_names(idx);
    
        best_features = sorted_feature_names(:, 1:3);   

        Energy_Best_3F(cellItem,:) = best_features;

   end
% Extract unique best features
Ebest = unique(Energy_Best_3F(:));
Ebest = Ebest(1:3,1);

% Display best features
disp('Best 3 features for Lagging Energy Data:');
disp(Ebest);

%% STEP 3: Create Lag Matrices for Energy

lags = [1];
lag_energy = cell(1,3);
   for item = 1:3
    table= energy{item};
    lag_matrix = lagmatrix(table, lags, 'DataVariables', ["Current Energy", "R2", "R8"]); %arbitrarily selected lag variables
    lag_mat = lag_matrix(:, 2:end);
    lag_mat = [table(:, 1:end-1), lag_mat(:, {'Lag1Current Energy', 'Lag1R2', 'Lag1R8'}), table(:, end)];
    lag_mat = rmmissing(lag_mat);
    lag_energy{item}=lag_mat;
    eval(['Tbl_E' num2str(item) '_lag = lag_mat;']);
   end
     
   
%% STEP 4: Create a Linear Model For Energy 1,2,3
NoEnergyTables = 3;

for i = 1:NoEnergyTables
    data = energy{i};
    Energy_model = fitlm(data(:, 2:end));
end

%% Create and Train Model for Lagged Data
NoEnergyTables = 3;

for i = 1:NoEnergyTables
    data = lag_energy{i};
    LEmodel = fitlm(data(:, 2:end));   
end

%% NRMSE for Energy 
% Energy Prediction
NRMSE_E = zeros(3,2);

for i = 1:3
data = table2array(energy{i});
data1 = table2array(lag_energy{i});

X_train = data(:,2:end-1);
y_train = data(:,end);
y_pred_NL = predict(Energy_model,X_train);
NRMSE_E(i,1) = 100*calculate_nrmse(y_train,y_pred_NL); % in percentage

X_trL = data1(:,2:end-1);
y_trL = data1(:,end);
y_pred_L = predict(LEmodel,X_trL);
NRMSE_E(i,2) = 100*calculate_nrmse(y_trL,y_pred_L); % in percentage

end
NRMSE_E = array2table(NRMSE_E, VariableNames={'M-NLE','M-LE'},RowNames={'E1','E2','E3'});

%% Visualize The NRMSE for Energy
MAE_E = table2array(NRMSE_E);

% Sort Avg_Error and get the corresponding indices
[sortedError, sortedIndices] = sort(MAE_E);

% Create a bar plot with sorted data
bar(sortedError);
% Customize the plot
ylabel('NRMSE');
title('NRMSE values Lagged vs NoN Lagged Data');
xticklabels({'E1', 'E2', 'E3'}); % Custom x-axis labels
legend('Lagged Data', 'Non Lagged Data'); % Add legend with custom labels
grid on;

%% SUMMARY
% Best Performing Model for energy
% M_LE

%% TESTING THE MODEL

Data1 = load("redhouseTest1.mat");
%Data2 = load("redhouseTest2.mat");

Data = Data1; % select the testing data set

proxy = Data.proxy;
test_t = Data.t;
test_u = Data.u;
test_x = Data.x;
test_y = Data.y;

%% GENERATE TESTING DATA

E_Col = {'time', 'Current Energy', 'R1','R2','R3','R4','R5', 'R6','R7','R8','R9','R10', 'Energy Outputs'};

TestE1 = [test_t', test_x(1,:)', test_x(4:13,:)',  test_y(1,:)'];
TestE1 = fillmissing(TestE1,"linear");

TestE2 = [test_t', test_x(2,:)', test_x(4:13,:)',  test_y(2,:)'];
TestE2 = fillmissing(TestE2,"linear");

TestE3 = [test_t', test_x(3,:)', test_x(4:13,:)',  test_y(3,:)'];
TestE3 = fillmissing(TestE3,"linear");

TestTbl_E1 = array2table(TestE1, 'VariableNames',E_Col);
TestTbl_E2 = array2table(TestE2, 'VariableNames',E_Col);
TestTbl_E3 = array2table(TestE3, 'VariableNames',E_Col);

% store tables in Cell arrays. 
TestEnergy = {TestTbl_E1,TestTbl_E2,TestTbl_E3};

%% GENERATE LAGGED ENERGY DATA
lags = [1];
Testlag_energy = cell(1,3);
   for item = 1:3
    table= TestEnergy{item};
    lag_matrix = lagmatrix(table, lags, 'DataVariables', ["Current Energy", "R2", "R8"]); %arbitrarily selected lag variables
    lag_mat = lag_matrix(:, 2:end);
    lag_mat = [table(:, 1:end-1), lag_mat(:, {'Lag1Current Energy', 'Lag1R2', 'Lag1R8'}), table(:, end)];
    lag_mat = rmmissing(lag_mat);
    Testlag_energy{item}=lag_mat;
   end

   %% PREDICT and FIND NRMSE

NRMSE_E_Test = zeros(3,2);

for i = 1:3
data = table2array(TestEnergy{i});
data1 = table2array(Testlag_energy{i});

X_test = data(:,2:end-1);
y_test = data(:,end);
y_pred_NL = predict(Energy_model,X_test);
NRMSE_E_Test(i,1) = 100*calculate_nrmse(y_test,y_pred_NL); % in percentage

X_tstL = data1(:,2:end-1);
y_tstL = data1(:,end);
y_pred_RF = predict(LEmodel,X_tstL);
NRMSE_E_Test(i,2) = 100*calculate_nrmse(y_tstL,y_pred_RF); % in percentage
end
NRMSE_E_Test = array2table(NRMSE_E_Test, VariableNames={'M-NLE','M-LE'},RowNames={'TestE1','TestE2','TestE3'});

%% Visualize The NRMSE for Energy
MAE_TestE = table2array(NRMSE_E_Test);

% Sort Avg_Error and get the corresponding indices
[sortedError, sortedIndices] = sort(MAE_TestE);

% Create a bar plot with sorted data
bar(sortedError);
% Customize the plot
ylabel('NRMSE');
title('NRMSE values Lagged vs NoN Lagged Data');
xticklabels({'TestE1', 'TestE2', 'TestE3'}); % Custom x-axis labels
legend('Lagged Data', 'Non Lagged Data'); % Add legend with custom labels
grid on;
%% 
 % save the plot to a file
saveas(gcf,'Energy NRMSE Comparison.png');
