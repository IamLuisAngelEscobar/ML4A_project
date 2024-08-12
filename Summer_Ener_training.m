%% PROJECT
% Training part

% University Of L'Aquila
% COURSE: Machine Learning For Automation

% PROJECT TITLE: 
% REDHOUSE, REGRESSION FOR STATE PREDICTION

% AUTHORS
% Luis Angel Escobar Hernandez
% Saidi Atwaya Nchimbi

%%
clear;
clc;
close all;

%% STEP 1: Load the data
% Load the data: Disturbance (d), Inputs (u), States (x), Outputs (y)

directory = '/Users/luisescobar/Downloads/Summer_attempt_2024/ML4A/Project';
filename = 'redhouseTrain.mat';
Train_Data = load(fullfile(directory, filename));
% Set the random number generator to default for reproducibility
rng("default")

d = Train_Data.d;
proxy = Train_Data.proxy;
t = Train_Data.t;
u = Train_Data.u;
x = Train_Data.x;
y = Train_Data.y;

%  Data Of Each Room

% Get Disturbance Data (same for each room)
ext_temp = d(1, :);
ext_humd = d(2, :);
w_chill = d(3, :);
h_chill = d(4, :);
press = d(5, :);
rain = d(6, :);
sol_rad = d(7, :);

% convert Proxy to date format.
% proxy contains month of the year, hour of the day, day of the week
% day of the week is in the format [1,2 ,..., 7]
% we are assuming 1 is monday while 7 is sunday

% We know the ending date and time from the proxy information
% Each hour of data is divided in 30 slots
% Thus, each reading is done each 2 minutes (30 slots * 2 min= 60 min)
% Up to this point we know month and the time

% For the date and year we can consider a whole month of data
% We need to consider in which date it starts [1, 2, ..., 7] and in which 
% date it ends
% From this information we can know to which year it belongs
% Then, I can start to add to the proxy the number of the day
% This will be my reference so I can enumerate the other months

end_date = datetime('2024-02-26 11:44:00');

% Define the time interval between data points (in minutes)
time_interval = 2;

% Define the number of rows in your time series
num_rows = 72369;

% Calculate the starting date based on the ending date, number of rows, and time interval
start_date = end_date - minutes((num_rows - 1) * time_interval);

% Generate the timestamps for each data point
timestamps = start_date + minutes(0:time_interval:(num_rows-1)*time_interval);
timestamps = timestamps';

% Display the first few timestamps to verify
disp(timestamps(1:3));
disp('STEP 1 --> completed')

%% STEP 2: Creata an array with the information of all the rooms and clean it.
% [Alternative collect the data of all the rooms in a single array
% u corresponds to the inputs or desired temperature in the room
% x corresponds to the states or current temperature in the room
% From disturbances I'll consider External Temp, External Humidity, Wind
% Chill, Heat Chill. We'll discard the last ones since the description
% indicates that they are not too important
%ColumnHeads = {'time', 'U', 'X', 'External Temp', 'External Humidity', 'Wind Chill', 'Heat Chill'};
% ColumnsHeads contains 27 variables
ColumnHeads_ener = {'E1', 'E2', 'E3'};
ColumnHeads = {'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', ...
    'U_1', 'U_2', 'U_3', 'U_4', 'U_5', 'U_6', 'U_7', 'U_8', 'U_9', 'U_10', ...
    'External Temp', 'External Humidity', 'Wind Chill', 'Heat Chill'};

% We create the feature matrix with the information of all the rooms
X_feat_ener = [x(1,:)', x(2,:)', x(3,:)'];
X_feat = [x(4:13,:)',  u', ext_temp', ext_humd', w_chill', h_chill'];

% Linear interpolation since it seems that we have some NaN values
X_feat_ener = fillmissing(X_feat_ener,"linear");
X_feat = fillmissing(X_feat,"linear");

Tbl_X_ener = array2table(X_feat_ener, 'VariableNames',ColumnHeads_ener);
Tbl_X = array2table(X_feat, 'VariableNames',ColumnHeads);

disp('STEP 2 --> completed')

%% STEP 3: Create lag matrices

%lag_rooms = cell(1,10);
numEnergy = 3;
% lags for 2,10,20,30,40,50,60 min
lags = [1 5 10 15 20 25 30];
cutoff = max(lags);
lag_matrix = lagmatrix(Tbl_X, lags);
lag_mat = lag_matrix(:, 2:end);

% We have a matrix with all the lagged values size 24*7 = 168 columns
lag_mat = rmmissing(lag_mat);

% For energy
lag_matrix_ener = lagmatrix(Tbl_X_ener, lags);
lag_mat_ener = lag_matrix_ener(:, 2:end);

% We have a matrix with all the lagged values size 24*7 = 168 columns
lag_mat_ener = rmmissing(lag_mat_ener);

disp('STEP 3 --> completed')

%% STEP 4: Train the models 

%% STEP 4.1: Linear regression
% to predict temperature
% So, for room1_lag I will have
% fitlm 2 min (:,1:24) corresponding to the variables --> [row_u', row_x', ext_temp', ext_humd', w_chill', h_chill']
% fitlm 10 min (:,25:48)
% fitlm 20 min (:,13:18)
% fitlm 30 min (:,19:24)
% fitlm 40 min (:,25:30)
% fitlm 50 min (:,31:36)
% fitlm 60 min (:,37:42)
% So I will have a total of 10*7= 70 models 
lmModelL = cell(3,7);
%numRooms = 10;
ModelNames = {'Mdl_2Min', 'Mdl_10Min', 'Mdl_20Min', 'Mdl_30Min','Mdl_40Min', 'Mdl_50Min', 'Mdl_60Min'}; 
energyNames = {'E1', 'E2', 'E3'};
% For outputs we are considering E1, E2 , E3
num_columns = 24;
for energyNumber = 1:numEnergy
    %Energy = eval(['energy' num2str(energyNumber) '_lag']);
    outputs = y(energyNumber, :)';
    outputs = fillmissing(outputs,"linear");
    %out_trunc = outputs(1:height(Room));
    out_trunc = outputs(cutoff+1:end);
    
    
    for modelNo = 1:7
        col_ener = energyNumber+(modelNo-1)*3;
        X_ener = lag_mat_ener(:,col_ener);
        X = lag_mat(:,(modelNo*num_columns)-(num_columns-1):modelNo*num_columns);     
        data = [X, table(out_trunc, 'VariableNames', {'Prediction'})];
        data = [X_ener, data];
        model = fitlm(data);
        
        % dynamically assign Model Names
        lmModelL{energyNumber,modelNo}=model;
    end    
end
lmModelL = cell2table(lmModelL,VariableNames=ModelNames,RowNames=energyNames);
disp('STEP 4.1 --> completed')


%% STEP 4.2: Random forest regression 

RFModelL = cell(3,7);
%numRooms = 10;
numTrees = 10;

for energyNumber = 1:numEnergy
    %Room = eval(['room' num2str(energyNumber) '_lag']);
    outputs = y(energyNumber, :)';
    outputs = fillmissing(outputs,"linear");
    out_trunc = outputs(cutoff+1:end);
    
    for modelNo = 1:7
        %X = Room(:,(modelNo*6)-5:modelNo*6);
        col_ener = energyNumber+(modelNo-1)*3;
        X_ener = lag_mat_ener(:,col_ener);
        X = lag_mat(:,(modelNo*num_columns)-(num_columns-1):modelNo*num_columns);     
        data = [X, table(out_trunc, 'VariableNames', {'Prediction'})];
        data = [X_ener, data];
        %data = [X, table(out_trunc, 'VariableNames', {'Prediction'})];
        model = TreeBagger(numTrees,data,out_trunc,Method="regression");
        
        % dynamically assign Model Names
        RFModelL{energyNumber,modelNo}=model;
    end
    disp(energyNumber)
end
RFModelL = cell2table(RFModelL,VariableNames=ModelNames,RowNames=energyNames);
disp('STEP 4.2 --> completed')

%% STEP 5: Computing NRMSE (training dataset)

%% STEP 5.1: For linear regression
NRMSE_lmodel = zeros(3,7);

for energyNumber = 1:numEnergy
    %Room = table2array(eval(['room' num2str(energyNumber) '_lag']));
    % Is this the same data as the one used for training the model? YES for
    % both variables
    %X = Room(:,2:end-7);
    %y_train = Room(:,end-6:end);
    outputs = y(energyNumber, :)';
    outputs = fillmissing(outputs,"linear");
    out_trunc = outputs(cutoff+1:end);
    
    for modelNo = 1:7
        % same situation here 
        %y_tr = y_train(:,modelNo);
        %X = Room(:,(modelNo*6)-5:modelNo*6); 
        col_ener = energyNumber+(modelNo-1)*3;
        X_ener = lag_mat_ener(:,col_ener);
        X = lag_mat(:,(modelNo*num_columns)-(num_columns-1):modelNo*num_columns);     
        data = [X, table(out_trunc, 'VariableNames', {'Prediction'})];
        data = [X_ener, data];
        %model = fitlm(data);
        % we load the model and we do the corresponding predictions
        % shouldn't we do this with the test dataset?
        lmdl = lmModelL{energyNumber,modelNo};
        lmodel = lmdl{1};
        y_pred = predict(lmodel,data);
        NRMSE_lmodel(energyNumber,modelNo) = 100*calculate_nrmse(out_trunc,y_pred);
    end    
end
NRMSE_lmodel = array2table(NRMSE_lmodel,"RowNames",energyNames,"VariableNames",ModelNames);
disp('STEP 5.1 --> completed')

%% STEP 5.2: For random forest regression
NRMSE_RF = zeros(3,7);

for energyNumber = 1:numEnergy
    %Room = table2array(eval(['room' num2str(roomNumber) '_lag']));
    %X = Room(:,2:end-7);
    %y_train = Room(:,end-6:end);
    outputs = y(energyNumber, :)';
    outputs = fillmissing(outputs,"linear");
    out_trunc = outputs(cutoff+1:end);
    
    for modelNo = 1:7
        %X = Room(:,(modelNo*6)-5:modelNo*6); 
        %y_tr = y_train(:,modelNo);
        col_ener = energyNumber+(modelNo-1)*3;
        X_ener = lag_mat_ener(:,col_ener);
        X = lag_mat(:,(modelNo*num_columns)-(num_columns-1):modelNo*num_columns);     
        data = [X, table(out_trunc, 'VariableNames', {'Prediction'})];
        data = [X_ener, data];
        RFmdl = RFModelL{energyNumber,modelNo};
        model = RFmdl{1};
        y_pred = predict(model,data);
        NRMSE_RF(energyNumber,modelNo) = 100*calculate_nrmse(out_trunc,y_pred);
    end    
end
NRMSE_RF = array2table(NRMSE_RF,"RowNames",energyNames,"VariableNames",ModelNames);
disp('STEP 5.2 --> completed')

%% STEP 6: Visualize the NRMSE for each room for the different cases
% Plot line graphs for each room
X_Axis = {'2 Min', '10 Min', '20 Min', '30 Min','40 Min', '50 Min', '60 Min'};

for energyNumber = 1:3
    % Extract errors for the current room from all three tables
    error_lm = table2array(NRMSE_lmodel(energyNumber, :));
    error_rf = table2array(NRMSE_RF(energyNumber, :));
    
    % Define the x-axis data
     
    X1 = 1:7;
    
    % Plot line graph for the current room
    figure;
    %plot(X1, error_lm_NoLag, 'o-', 'LineWidth', 2);  
    %hold on;
    plot(X1, error_lm, 'o-', 'LineWidth', 2);
    hold on
    plot(X1, error_rf, 'o-', 'LineWidth', 2);
    hold off;
    
    % Add title and labels
    title(['Energy ', num2str(energyNumber), ' Error Comparison']);
    xlabel('Time Predictions');
    xticklabels(X_Axis);
    ylabel('Error Value');
    %legend('Linear Reg No lag', 'Lagged Linear Reg', 'Random Forest Reg');
    legend('Linear Reg', 'Random Forest Reg');
    
    grid on;
    %break
    
end
disp('STEP 6 --> completed')

%% STEP 7: Save the models for later testing

fullpath_lmodel = (fullfile(directory, 'lmModelL_energy.mat'));
save(fullpath_lmodel, 'lmModelL');

fullpath_lmodel = (fullfile(directory, 'RFModelL_energy.mat'));
save(fullpath_lmodel, 'RFModelL');
disp('STEP 7 --> completed')
disp('The models are ready to use')

%% RESULT SUMMARY

% Linear regressor perfomed the best
% Why this happens?
% We can try to increase the number of trees. Currently we are working only
% with 10