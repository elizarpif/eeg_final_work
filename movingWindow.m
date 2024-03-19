EEG = load('EEG3.mat');
EEG3 = EEG.EEG';

fnyquist = 256.160664;
fs = fnyquist * 2;
ts = 1 / fs;

% New sampling and Nyquist frequencies
fnyquist_new = fnyquist / 2;
fs_new = fnyquist_new * 2;

% Define the channels to process
% [28, 51, 19, 1, 35, 30];
% channelsToProcess = [28, 51, 19, 1, 35, 30];
channelsToProcess = 1:64;

%% Parameters for nonlinear prediction error
embedding_dim = 6;
num_neighbors = 3;
theiler_correction = 30;
prediction_horizon = 2;
time_delays = 19; %1:3:49;
tau = 19;
numTimeDelays = length(time_delays);

%% parallel workers setup
poolObj = gcp('nocreate'); % Get the current pool without creating a new one
if isempty(poolObj)
    poolObj = parpool; % If no pool, start the default pool
end
numWorkers = poolObj.NumWorkers;
fprintf('Number of workers in the current pool: %d\n', numWorkers);

% Preallocate results matrix for all channels
prediction_error_all_channels = zeros(length(channelsToProcess), numel(downsampledFilteredSignalSegments) * 2, 1); %numTimeDelays);

%% Loop over the selected channels
parfor chIdx = 1:length(channelsToProcess)
    channel = channelsToProcess(chIdx);
    disp(['Processing Channel: ', num2str(channel)]);
    
    %% WINDOW SELECTION
    % Adjust for each channel - selection of windows: 539s is the whole signal timing
    signal = EEG3(channel, :);
    
    totalTime = 538; % Total duration in seconds
    segmentDuration = 20; % Duration of each segment in seconds
    numSegments = fix(totalTime / segmentDuration); % Number of segments to extract
    
    % Preallocate the cell array for segments
    signalSegments = cell(1, numSegments);
    
    startTime = 1;
    for i = 1:numSegments
        endTime = startTime + segmentDuration - 1;
        signalSegments{i} = signal(round(startTime/ts):round(endTime/ts));
        startTime = startTime + segmentDuration;
    end
    
    %% BUTTERWORTH FILTER
    fcutoff = fnyquist_new * 0.9;
    order = 8;
    [b, a] = butter(order, fcutoff / fnyquist_new, 'low');
    
    % Filter segments
    filteredSignalSegments = cellfun(@(x) filter(b, a, x), signalSegments, 'UniformOutput', false);
    
    %% DOWNSAMPLING
    downsampledFilteredSignalSegments = cellfun(@(x) x(1:2:end), filteredSignalSegments, 'UniformOutput', false);
    
    %% SURROGATES
    numSignals = numel(downsampledFilteredSignalSegments);
    surrogates_matrix = zeros(numSignals * 2, length(downsampledFilteredSignalSegments{1}));
    
    for i = 1:numSignals
        surrogate_index = (i - 1) * 2 + 2;
        % Assuming phase_randomized_surrogate_aulaglobal is defined elsewhere
        surrogates_matrix(surrogate_index, :) = phase_randomized_surrogate_aulaglobal(downsampledFilteredSignalSegments{i});
        original_signal_index = (i - 1) * 2 + 1;
        surrogates_matrix(original_signal_index, :) = downsampledFilteredSignalSegments{i};
    end
    
    %% NONLINEAR PREDICTION ERROR
    prediction_error = zeros(size(surrogates_matrix, 1), 1);
    
    currentDateTime = datetime;
    disp(['Current date and time before execution: ', char(currentDateTime)]);
    %% Parallel Processing Setup
    % Assuming gcp('nocreate') and related setup is done outside the loop
    %parfor segmentIdx = 1:numSegments
    tic;
       
    tempPredictionError = zeros(size(surrogates_matrix, 1), 1);
    
    for signal_idx = 1:size(surrogates_matrix, 1)
        tempPredictionError(signal_idx) = CalculateError(surrogates_matrix(signal_idx,:), embedding_dim, tau, num_neighbors, theiler_correction, prediction_horizon);
    end
    
    prediction_error = tempPredictionError;
    
    elapsedTime = toc;
    fprintf('Channel %d, execution time: %.2f seconds.\n', channel, elapsedTime);
    %end
    
    %% Store the results
    prediction_error_all_channels(chIdx, :, :) = prediction_error;

    currentDateTime = datetime;
    disp(['Current date and time after execution: ', char(currentDateTime)]);
end

disp('All channels processed.');

%% VISUALISE RESULTS

% Initialize the diff_matrix with zeros
% initialize a matrix
% that has 1 row and 26 columns to hold the differences for each pair.
% Assuming 'prediction_error' has errors for 6 channels, and you're interested in some form of difference calculation

numChannels = length(channelsToProcess); % Number of channels
numComparisons = size(prediction_error, 1) / 2; % Assuming comparison between successive columns
diff_matrix = zeros(numChannels, numComparisons); % Adjusted for a difference per channel for each comparison

% Loop over each channel
for ch = 1:numChannels
    % Calculate differences between successive elements for each channel
    for col = 1:numComparisons
        diff_matrix(ch, col) = abs(prediction_error_all_channels(ch, col * 2)-prediction_error_all_channels(ch, col *2 -1));
    end
end

% Plot the matrix
figure;
imagesc(diff_matrix);
title('Difference Matrix of Signal and His Surrogate');
colorbar;

