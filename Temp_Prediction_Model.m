%% PROJECT

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

%% STEP 1: Load the Data
% Load the data: Disturbance (d), Inputs (u), States (x), Outputs (y)

directory = '/Users/luisescobar/Downloads/Summer_attempt_2024/ML4A/Project';
filename = 'redhouseTrain.mat';
Train_Data = load(fullfile(directory, filename));

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

%%  STEP 1.1: Define the lags values
% We can try to compute the ACF [autocorrelation function, WQU 033] to see 
% until which lag value the prediction power is good enough
futureOutNames = {'TenMin', 'TwentyMin', 'ThirtyMin','FortyMin', 'FiftyMin', 'SixtyMin'};
N_row = size(x,2);
N_column = size(futureOutNames,2);
future_outputs = NaN(N_row,N_column);

%% STEP 2: Collect Data of Each Room into an array and clean it.
% u corresponds to the inputs or desired temperature in the room
% x corresponds to the states or current temperature in the room
ColumnHeads = {'time', 'U', 'X', 'External Temp', 'External Humidity', 'Wind Chill', 'Heat Chill', 'Pressure', 'Rain', 'Solar Radiation','TwoMin', 'TenMin', 'TwentyMin', 'ThirtyMin','FortyMin', 'FiftyMin', 'SixtyMin'};
rooms = cell(1, 10);
% we iterate along the different rooms
for i = 1:10
        % input of room i
        row_u = u(i, :);
        % state of room i. The first 3 rows of array x correspond to 
        % energy values, therefore we must use i+3
        row_x = x(i + 3, :);
        % same situation for the outputs
        outputs = y(i + 3, :)';
        outputs = fillmissing(outputs,"linear");

% data sampling time was 2 mins : 
% we select indices 2,5,10,15,20,25,30 of the output to make
% 2,10,20,30,40,50,60 min predictions of sample 1 of the input.

% We are considering data just after the jump
% We have the following jumps 5*1, 5*2, 5*3, 5*4, 5*5, 5*6
% Since data is sampled each 2 minutes, this jumps becomes
% 5*1*2, 5*2*2, 5*3*2, 5*4*2, 5*5*2, 5*6*2 corresponding to the minutes of interest
% 10 min, 20 min, 30 min, 40 min, 50 min, 60 min

         for j = 1:N_column 
             jump = 5*j ;
             future_outputs(1:end-jump,j)= outputs(jump+1:end,:); 
         end
% future output 1 data after the first 10 min
% future output 2 data after the first 20 min
% future output 3 data after the first 30 min
% future output 4 data after the first 40 min
% future output 5 data after the first 50 min
% future output 6 data after the first 60 min
        room =  [t', row_u', row_x', ext_temp', ext_humd', w_chill', h_chill', press', rain', sol_rad',outputs];
        
        % fill missing values by linear interpolation.
        room = fillmissing2(room,"linear");
        % concatenate future_outputs matrix to each room.
        room = [room,future_outputs];
        % fill  cell array
        rooms{i} = rmmissing(array2table(room,VariableNames=ColumnHeads));
        % assign table names to each dataset in the cell array
        eval(['room' num2str(i) ' = rooms{i};'])
       
end

% We can try a different approach for future_outputs
% Instead of going to the future
% future_outputs(1:end-jump,j)= outputs(jump+1:end,:);
% We can go to the past
% future_outputs(1:end-jump,j)= outputs(jump+1:end,:);
%% STEP 3: Feature Scoring For Each Room
% We are evaluating a fitlm model for each room using Data which contains:
% [time, U, X, External Temp, External Humidity, Wind Chill, Heat Chill,
% Preassure, Rain, Solar Radiation, TwoMin lag (future)] to obtain the top 
% 3 features

NSD_Best_3F = cell(10,3);
    for roomNumber = 1:10
        Data = eval(['room' num2str(roomNumber)]);
        Data = Data(:,1:end-6);
        fs_model = fitlm(Data);
        coefficients = table2array(fs_model.Coefficients(:, 'Estimate'));
        feature_names = Data.Properties.VariableNames;
        [~, idx] = sort(abs(coefficients(2:end)), 'descend');
        sorted_coefficients = coefficients(idx + 1);
        sorted_feature_names = feature_names(idx);
        best_features = sorted_feature_names(:, 1:3);

        % KEY: NSD_Best_3F = Non Standardized Data Best 3 Features

        NSD_Best_3F(roomNumber,:) = best_features;

   end
% Extract unique best features
best = unique(NSD_Best_3F(:));

% Display best features
disp('Best 3 features for NON STANDARDIZED DATA:');
disp(best);
  
% QUESTION
% Should we standardize data for Time Series prediction? 
%% STEP 4: Create Lag Matrices

lag_rooms = cell(1,10);
numRooms = 10;
lags = [1 2];

for roomNumber = 1:numRooms
    currentRoom = eval(['room' num2str(roomNumber)]);
    % We are creating a lag matrix for roomx which additionally to 
    % [time, U, X, External Temp, External Humidity, Wind Chill, Heat Chill,
    % Preassure, Rain, Solar Radiation, TwoMin lag (future)] includes the
    % variables TenMin, TwentyMin, ThirtyMin, FortyMin, FiftyMin, SixtyMin
    % Consider that we are creating a lag matrix according to the
    % documentation. In this case we'll have -2 min and -4 min information
    % we create lag matrix just for u, x, rain since are the top 3 features
    lag_matrix = lagmatrix(currentRoom, lags, 'DataVariables', ["U", "X", "Rain"]);
    % we omit the time column
    lag_mat = lag_matrix(:, 2:end);
    % we create a final lag matrix concatenating 
    % first 10 columns of roomx data + top 3 features lagged +  last 7
    % columns of roomx data; corresponding to the lagged temperatura states
    lag_mat = [currentRoom(:, 1:end-7), lag_mat(:, {'Lag1Rain', 'Lag1U', 'Lag1X', 'Lag2Rain', 'Lag2U', 'Lag2X'}), currentRoom(:, end-6:end)];
    % remove missing entries from lag_mat
    lag_mat = rmmissing(lag_mat);
    lag_rooms{roomNumber} = lag_mat;
    % we save this new table as a roomx_lag
    eval(['room' num2str(roomNumber) '_lag = lag_mat;']);
end

% QUESTION
% If we are creating lag matrices why did we create future_outputs in Step
% 2?
% In future_outputs we create columns which is part of the table roomx
% for our target prediction values. Now we are creating a lag matrix which
% implies that we are shifting our data again. What is the consequence of
% doing this?
% Aren't the last values [] of this lag matrix wrong?

% Shouldn't we directly create here our lag matrices just with the top 3
% feature variables and the temperature of the room
% so, we'll have
% lags = [1 5 10 15 20 25 30]
% for lags at 2 min, 10 min, 20 min, 30 min, 40 min, 50 min, 60 min
% Remember 
% we want to predict the energy consumption

%% STEP 5: Train The Models

lmModelNL = cell(10,7);
numRooms = 10;
ModelNames = {'Mdl_2Min', 'Mdl_10Min', 'Mdl_20Min', 'Mdl_30Min','Mdl_40Min', 'Mdl_50Min', 'Mdl_60Min'}; 
roomNames = {'room1', 'room2', 'room3', 'room4','room5','room6','room7', 'room8','room9','room10'};

for roomNumber = 1:numRooms
    Room = eval(['room' num2str(roomNumber)]);
    X = Room(:,2:end-7);
    y_train = Room(:,end-6:end);
    
    for modelNo = 1:7
        y = y_train(:,modelNo);
        data = [X,y];
        model = fitlm(data);
        
        % dynamically assign Model Names
        lmModelNL{roomNumber,modelNo}=model;
    end    
end
lmModelNL = cell2table(lmModelNL,VariableNames=ModelNames,RowNames=roomNames);

%% Lagged Data Models

%% LINEAR RESGRESSION

lmModelL = cell(10,7);
numRooms = 10;
ModelNames = {'Mdl_2Min', 'Mdl_10Min', 'Mdl_20Min', 'Mdl_30Min','Mdl_40Min', 'Mdl_50Min', 'Mdl_60Min'}; 
roomNames = {'room1', 'room2', 'room3', 'room4','room5','room6','room7', 'room8','room9','room10'};

for roomNumber = 1:numRooms
    Room = eval(['room' num2str(roomNumber) '_lag']);
    X = Room(:,2:end-7);
    y_train = Room(:,end-6:end);
    
    for modelNo = 1:7
        y = y_train(:,modelNo);
        data = [X,y];
        model = fitlm(data);
        
        % dynamically assign Model Names
        lmModelL{roomNumber,modelNo}=model;
    end    
end
lmModelL = cell2table(lmModelL,VariableNames=ModelNames,RowNames=roomNames);

% In my opinion we don't need two linear regression model. 
% One with top 3 lag features and one without top 3 lag features. 
% We just need a single matrix in which we include 
% (in this code we are including the top 3 features + all the other features when
% training the model; this does not make sense)
% time / U (input) / X (state) / top 3 features
%   3      26           25
% and from this matrix we can obtain the lag matrices
% when evaluating the top-3 features we can discard U and X since are
% variables that must be part of the DataFrame. We can consider just the
% top 3 disturbances
% We are training each linear model independently; one model for each room
% Is there a way to fix this?---> VAR
% Instead of choosing arbitrary the lag times, we can use an Autocorrelation function
% to see up to which lag value we have prediction power

% This code predicts the energy consumption [E1, E2, E3] just for one room 
%% RANDOM FOREST REGRESSOR

RFModelL = cell(10,7);
numRooms = 10;
numTrees = 10;

for roomNumber = 1:numRooms
    Room = eval(['room' num2str(roomNumber) '_lag']);
    X = Room(:,2:end-7);
    y_train = Room(:,end-6:end);
    
    for modelNo = 1:7
        y = y_train(:,modelNo);
        data = [X,y];
        model = TreeBagger(numTrees,X,y,Method="regression");
        
        % dynamically assign Model Names
        RFModelL{roomNumber,modelNo}=model;
    end    
end
RFModelL = cell2table(RFModelL,VariableNames=ModelNames,RowNames=roomNames);




%% Computing NRMSE for all rooms Unlagged Data.

% initiate matrix to store NRMSE for 7 Cases.
NRMSE_NL = zeros(10,7);

for roomNumber = 1:numRooms
    Room = table2array(eval(['room' num2str(roomNumber)]));
    X = Room(:,2:end-7);
    y_train = Room(:,end-6:end);
    
    for modelNo = 1:7
        y_tr = y_train(:,modelNo);
        mdl = lmModelNL{roomNumber,modelNo};
        model = mdl{1};
        y_pred = predict(model,X);
        NRMSE_NL(roomNumber,modelNo) = 100*calculate_nrmse(y_tr,y_pred);
    end    
end
NRMSE_NL= array2table(NRMSE_NL,"RowNames",roomNames,"VariableNames",ModelNames);

%% Computing NRMSE for Lagged Data Models.
% For Lags
NRMSE_lmodel = zeros(10,7);

for roomNumber = 1:numRooms
    Room = table2array(eval(['room' num2str(roomNumber) '_lag']));
    X = Room(:,2:end-7);
    y_train = Room(:,end-6:end);
    
    for modelNo = 1:7
        y_tr = y_train(:,modelNo);
        lmdl = lmModelL{roomNumber,modelNo};
        lmodel = lmdl{1};
        y_pred = predict(lmodel,X);
        NRMSE_lmodel(roomNumber,modelNo) = 100*calculate_nrmse(y_tr,y_pred);
    end    
end
NRMSE_lmodel = array2table(NRMSE_lmodel,"RowNames",roomNames,"VariableNames",ModelNames);

%% RF Error 

% For Lags
NRMSE_RF = zeros(10,7);

for roomNumber = 1:numRooms
    Room = table2array(eval(['room' num2str(roomNumber) '_lag']));
    X = Room(:,2:end-7);
    y_train = Room(:,end-6:end);
    
    for modelNo = 1:7
        y_tr = y_train(:,modelNo);
        RFmdl = RFModelL{roomNumber,modelNo};
        model = RFmdl{1};
        y_pred = predict(model,X);
        NRMSE_RF(roomNumber,modelNo) = 100*calculate_nrmse(y_tr,y_pred);
    end    
end
NRMSE_RF = array2table(NRMSE_RF,"RowNames",roomNames,"VariableNames",ModelNames);


%% STEP 9: Visualize the MSE for each room for the different cases
% Plot line graphs for each room
X_Axis = {'2 Min', '10 Min', '20 Min', '30 Min','40 Min', '50 Min', '60 Min'};

for roomNumber = 1:10
    % Extract errors for the current room from all three tables
    error_lm_NoLag = table2array(NRMSE_NL(roomNumber, :));
    error_lm = table2array(NRMSE_lmodel(roomNumber, :));
    error_rf = table2array(NRMSE_RF(roomNumber, :));
    
    % Define the x-axis data
     
    X1 = 1:7;
    
    % Plot line graph for the current room
    figure;
    plot(X1, error_lm_NoLag, 'o-', 'LineWidth', 2);  
    hold on;
    plot(X1, error_lm, 'o-', 'LineWidth', 2);
    plot(X1, error_rf, 'o-', 'LineWidth', 2);
    hold off;
    
    % Add title and labels
    title(['Room ', num2str(roomNumber), ' Error Comparison']);
    xlabel('Time Predictions');
    xticklabels(X_Axis);
    ylabel('Error Value');
    legend('Linear Reg No lag', 'Lagged Linear Reg', 'Random Forest Reg');
    
    grid on;
    
end


%% TESTING THE MODEL

%% CREATE TESTING DATASETS

Data1 = load("redhouseTest1.mat");
Data2 = load("redhouseTest2.mat");

Data = Data1; % select the testing data set

d = Data.d;
proxy = Data.proxy;
t = Data.t;
u = Data.u;
x = Data.x;
y = Data.y;

%  Data Of Each Room

% Get Disturbance Data (same for each room)
ext_temp = d(1, :);
ext_humd = d(2, :);
w_chill = d(3, :);
h_chill = d(4, :);
press = d(5, :);
rain = d(6, :);
sol_rad = d(7,:); 

%% CREATE ROOM DATASETS

futureOutNames = {'TenMin', 'TwentyMin', 'ThirtyMin','FortyMin', 'FiftyMin', 'SixtyMin'};
N_row = size(x,2);
N_column = size(futureOutNames,2);
future_outputs = NaN(N_row,N_column);
%%
ColumnHeads = {'time', 'U', 'X', 'External Temp', 'External Humidity', 'Wind Chill', 'Heat Chill', 'Pressure', 'Rain', 'Solar Radiation','TwoMin', 'TenMin', 'TwentyMin', 'ThirtyMin','FortyMin', 'FiftyMin', 'SixtyMin'};

TestRooms = cell(1, 10);

for i = 1:10
        row_u = u(i, :);
        row_x = x(i + 3, :);
        outputs =y(i + 3, :)';
        outputs = fillmissing(outputs,"linear");

         for j = 1:N_column
             jump = 5*j + 1;
             future_outputs(1:end-jump+1,j)= outputs(jump:end,:); 
         end
        room =  [t', row_u', row_x', ext_temp', ext_humd', w_chill', h_chill', press', rain', sol_rad',outputs];
        
        % fill missing values by interpolation.
        room = fillmissing2(room,"linear");
        % concatinate future_outputs matrix to each room.
        room = [room,future_outputs];
        room = room(2:end,:);
        % fill up the cell array
        TestRooms{i} = rmmissing(array2table(room,VariableNames=ColumnHeads));
        % assign table names to each dataset in the cell array
        eval(['TestRoom' num2str(i) ' = TestRooms{i};']);
end

%% CREATE LAG MATRIX

TestLagRooms = cell(1,10);
numRooms = 10;
lags = [1 2];

for roomNumber = 1:numRooms
    currentRoom = eval(['TestRoom' num2str(roomNumber)]);
    lag_matrix = lagmatrix(currentRoom, lags, 'DataVariables', ["U", "X", "Rain"]);
    lag_mat = lag_matrix(:, 2:end);
    lag_mat = [currentRoom(:, 1:end-7), lag_mat(:, {'Lag1Rain', 'Lag1U', 'Lag1X', 'Lag2Rain', 'Lag2U', 'Lag2X'}), currentRoom(:, end-6:end)];
    lag_mat = rmmissing(lag_mat);
    TestLagRooms{roomNumber} = lag_mat;
    eval(['TestRoom' num2str(roomNumber) '_lag = lag_mat;']);
end

%% PREDICT Output and FIND NRMSE
% initiate matrix to store NRMSE for 7 Cases.
NRMSE_NLTest = zeros(10,7);

for roomNumber = 1:numRooms
    Room = table2array(eval(['TestRoom' num2str(roomNumber)]));
    X_test = Room(:,2:end-7);
    y_test = Room(:,end-6:end);
    
    for modelNo = 1:7
        y_tst = y_test(:,modelNo);
        mdl = lmModelNL{roomNumber,modelNo};
        model = mdl{1};
        y_pred = predict(model,X_test);
        NRMSE_NLTest(roomNumber,modelNo) = 100*calculate_nrmse(y_tst,y_pred);
    end    
end
NRMSE_NLTest = array2table(NRMSE_NLTest,"RowNames",roomNames,"VariableNames",ModelNames);

%% NRMSE for Unlagged Data.
% For Lags
NRMSE_LTest = zeros(10,7);

for roomNumber = 1:numRooms
    Room = table2array(eval(['TestRoom' num2str(roomNumber) '_lag']));
    X_test = Room(:,2:end-7);
    y_test = Room(:,end-6:end);
    
    for modelNo = 1:7
        y_tst = y_test(:,modelNo);
        lmdl = lmModelL{roomNumber,modelNo};
        model = lmdl{1};
        y_pred = predict(model,X_test);
        NRMSE_LTest(roomNumber,modelNo) = 100*calculate_nrmse(y_tst,y_pred);
    end    
end
NRMSE_LTest = array2table(NRMSE_LTest,"RowNames",roomNames,"VariableNames",ModelNames);

%% RANDOM FOREST REGRESSOR

NRMSE_RFTest = zeros(10,7);

for roomNumber = 1:numRooms
    Room = table2array(eval(['room' num2str(roomNumber) '_lag']));
    X_test = Room(:,2:end-7);
    y_test = Room(:,end-6:end);
    
    for modelNo = 1:7
        y_ts = y_test(:,modelNo);
        RFmdl = RFModelL{roomNumber,modelNo};
        model = RFmdl{1};
        y_pred = predict(model,X_test);
        NRMSE_RFTest(roomNumber,modelNo) = 100*calculate_nrmse(y_ts,y_pred);
    end    
end
NRMSE_RFTest = array2table(NRMSE_RFTest,"RowNames",roomNames,"VariableNames",ModelNames);


%% Visualize the MAE for each room for the different cases
% Plot line graphs for each room
X_Axis = {'2 Min', '10 Min', '20 Min', '30 Min','40 Min', '50 Min', '60 Min'};

for roomNumber = 1:10
    % Extract errors for the current room from all three tables
    error_lm_NoLag = NRMSE_NL(roomNumber, :);
    error_lm = NRMSE_lmodel(roomNumber, :);
    error_rf = NRMSE_RFTest(roomNumber, :);
    
    % Define the x-axis data
     
    X1 = 1:7;
    
    % Plot line graph for the current room
    figure;
    plot(X1, table2array(error_lm_NoLag), 'o-', 'LineWidth', 2);  
    hold on;
    plot(X1, table2array(error_lm), 'o-', 'LineWidth', 2);
    plot(X1, table2array(error_rf), 'o-', 'LineWidth', 2);
    hold off;
    
    % Add title and labels
    title(['TestRoom ', num2str(roomNumber), ' Error Comparison']);
    xlabel('Time Predictions');
    xticklabels(X_Axis);
    ylabel('Error Value');
    legend('Linear Reg No lag', 'Lagged Linear Reg', 'Random Forest Reg');
    
    grid on;
    
    % Optionally, save the plot to a file
    saveas(gcf, ['TestRoom', num2str(roomNumber), '_ErrorComparison.png']);
end

%% RESULT SUMMARY

% RF regressor perfomed the best
% lagging our data improved the error convergence as seen between the linear models


