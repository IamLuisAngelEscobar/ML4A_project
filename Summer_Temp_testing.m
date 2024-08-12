%% PROJECT
% Testing part

% University Of L'Aquila
% COURSE: Machine Learning For Automation

% PROJECT TITLE: 
% REDHOUSE, REGRESSION FOR STATE PREDICTION

% AUTHORS
% Luis Angel Escobar Hernandez
% Saidi Atwaya Nchimbi

% Running for training data uncomment lines 26 128 158
% Running for training data uncomment lines 25 127 157
%%
clear;
clc;
close all;

%% STEP 1: Load the data
% Load the data: Disturbance (d), Inputs (u), States (x), Outputs (y)

directory = '/Users/luisescobar/Downloads/Summer_attempt_2024/ML4A/Project';
filename1 = 'redhouseTest1.mat';
%filename1 = 'redhouseTrain.mat';
filename2 = 'redhouseTest2.mat';

Data1 = load(fullfile(directory, filename1));
Data2 = load(fullfile(directory, filename2));
%Select the test DataSet
Data = Data1;

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
sol_rad = d(7, :);


disp('STEP 1 --> completed')

%% STEP 2: Collect data of each room into an array and clean it.

ColumnHeads = {'U', 'X', 'External Temp', 'External Humidity', 'Wind Chill', 'Heat Chill'};

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
        
        %room =  [t', row_u', row_x', ext_temp', ext_humd', w_chill', h_chill'];
        room =  [row_u', row_x', ext_temp', ext_humd', w_chill', h_chill'];
        % fill missing values by linear interpolation.
        room = fillmissing2(room,"linear");
        % fill  cell array
        rooms{i} = rmmissing(array2table(room,VariableNames=ColumnHeads));
        % assign table names to each dataset in the cell array
        eval(['room' num2str(i) ' = rooms{i};'])
        disp(i)
       
end
disp('STEP 2 --> completed')
%% STEP 3: Create lag matrices

lag_rooms = cell(1,10);
numRooms = 10;
% lags for 2,10,20,30,40,50,60 min
lags = [1 5 10 15 20 25 30];
cutoff = max(lags);
for roomNumber = 1:numRooms
    currentRoom = eval(['room' num2str(roomNumber)]);

    lag_matrix = lagmatrix(currentRoom, lags);
    %we omit the time column
    lag_mat = lag_matrix(:, 2:end);
    % we create a final lag matrix concatenating 
    % first 10 columns of roomx data + top 3 features lagged +  last 7
    % columns of roomx data; corresponding to the lagged temperatura states
    %lag_mat = [currentRoom(:, 1:end-7), lag_mat(:, {'Lag1Rain', 'Lag1U', 'Lag1X', 'Lag2Rain', 'Lag2U', 'Lag2X'}), currentRoom(:, end-6:end)];
    % remove missing entries (the entire row) from lag_mat
    lag_mat = rmmissing(lag_mat);
    lag_rooms{roomNumber} = lag_mat;
    % we save this new table as a roomx_lag
    eval(['room' num2str(roomNumber) '_lag = lag_mat;']);
end
disp('STEP 3 --> completed')

%% STEP 4: Computing NRMSE (testing dataset)
ModelNames = {'Mdl_2Min', 'Mdl_10Min', 'Mdl_20Min', 'Mdl_30Min','Mdl_40Min', 'Mdl_50Min', 'Mdl_60Min'}; 
roomNames = {'room1', 'room2', 'room3', 'room4','room5','room6','room7', 'room8','room9','room10'};

%% STEP 4.1: For linear regression
NRMSE_lmodel = zeros(10,7);

% Load the model
datalm = load(fullfile(directory, 'lmModelL_temp.mat'));
lmModelL = datalm.lmModelL;

for roomNumber = 1:numRooms
    Room = table2array(eval(['room' num2str(roomNumber) '_lag']));
    % Is this the same data as the one used for training the model? YES for
    % both variables
    %X = Room(:,2:end-7);
    %y_train = Room(:,end-6:end);
    outputs = y(roomNumber + 3, :)';
    outputs = fillmissing(outputs,"linear");
    out_trunc = outputs(cutoff+1:end-1);
    %out_trunc = outputs(cutoff+1:end);
    
    for modelNo = 1:7
        % same situation here 
        %y_tr = y_train(:,modelNo);
        X = Room(:,(modelNo*6)-5:modelNo*6); 
        % we load the model and we do the corresponding predictions
        % shouldn't we do this with the test dataset?
        lmdl = lmModelL{roomNumber,modelNo};
        lmodel = lmdl{1};
        y_pred = predict(lmodel,X);
        NRMSE_lmodel(roomNumber,modelNo) = 100*calculate_nrmse(out_trunc,y_pred);
    end    
end
NRMSE_lmodel = array2table(NRMSE_lmodel,"RowNames",roomNames,"VariableNames",ModelNames);
disp('STEP 4.1 --> completed')
%% STEP 4.2: For random forest regression
NRMSE_RF = zeros(10,7);

% Load the model
datarf = load(fullfile(directory, 'RFModelL_temp.mat'));
RFModelL = datarf.RFModelL;

for roomNumber = 1:numRooms
    Room = table2array(eval(['room' num2str(roomNumber) '_lag']));
    %X = Room(:,2:end-7);
    %y_train = Room(:,end-6:end);
    outputs = y(roomNumber + 3, :)';
    outputs = fillmissing(outputs,"linear");
    out_trunc = outputs(cutoff+1:end-1);
    %out_trunc = outputs(cutoff+1:end);
    
    for modelNo = 1:7
        X = Room(:,(modelNo*6)-5:modelNo*6); 
        %y_tr = y_train(:,modelNo);
        RFmdl = RFModelL{roomNumber,modelNo};
        model = RFmdl{1};
        y_pred = predict(model,X);
        NRMSE_RF(roomNumber,modelNo) = 100*calculate_nrmse(out_trunc,y_pred);
    end
    disp(roomNumber)
end
NRMSE_RF = array2table(NRMSE_RF,"RowNames",roomNames,"VariableNames",ModelNames);
disp('STEP 4.2 --> completed')

%% STEP 5: Visualize the NRMSE for each room for the different cases% Plot line graphs for each room

X_Axis = {'2 Min', '10 Min', '20 Min', '30 Min','40 Min', '50 Min', '60 Min'};

for roomNumber = 1:10
    % Extract errors for the current room from all three tables
    %error_lm_NoLag = NRMSE_NL(roomNumber, :);
    error_lm = NRMSE_lmodel(roomNumber, :);
    error_rf = NRMSE_RF(roomNumber, :);
    
    % Define the x-axis data
     
    X1 = 1:7;
    
    % Plot line graph for the current room
    figure;
    %plot(X1, table2array(error_lm_NoLag), 'o-', 'LineWidth', 2);  
    %hold on;
    plot(X1, table2array(error_lm), 'o-', 'LineWidth', 2);
    hold on
    plot(X1, table2array(error_rf), 'o-', 'LineWidth', 2);
    hold off;
    
    % Add title and labels
    title(['TestRoom ', num2str(roomNumber), ' Error Comparison']);
    xlabel('Time Predictions');
    xticklabels(X_Axis);
    ylabel('Error Value');
    legend('Linear Reg', 'Random Forest Reg');
    
    grid on;
    %break
    % Optionally, save the plot to a file
    %saveas(gcf, ['TestRoom', num2str(roomNumber), '_ErrorComparison.png']);
end
%%