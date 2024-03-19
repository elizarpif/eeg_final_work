% Select 6-8 exemplary segments of individual channels 
% from you EEG recordings (Single channels,
% windows of 20 seconds. 
% Given the sampling rate, this corresponds to 10240 samples. 
% Prior to the
% analysis, down-sample your signals by a factor of 2, i.e., 
% take only every second sample. The resulting
% signal length of 5120 samples will be much faster to analyze.). 
% Your exemplary signals should include
% the different types of activity we saw, e.g., 
% recordings with or without epileptiform activity, with or
% without seizures, with or without artifacts. 
% Start to analyze them with the nonlinear prediction error
% and surrogates. 


global amplitude_parameter;
amplitude_parameter = 10; % Initial value

data = load("EEG3.mat");

eegData = data.EEG;
channelNameArray = data.channelNameArray;

eegDataT = eegData.';

% sampling frequency
Fs = (50/0.195221);

total_duration = length(eegDataT(1,:))/Fs;

% samples per 2 second
Ts = 1/Fs; 

time_vector = 0:Ts:total_duration;

eeg_data_by_two = eegDataT(30, 1:2:end);


% Extract the EEG data for the specified time interval
% eeg_data_interval = eegDataT(1, :);

signal_in = eeg_data_by_two(26/Ts:46/Ts);
signal_out = surrogates(signal_in);
signal_out2 = surrogates(signal_in);
% figure(1);
% plot(signal_out)
% figure(2);
% plot(signal_in)

% time vector less than eegData
figure(1);
plot_and_keypress(time_vector(1:length(signal_out)),signal_out,channelNameArray(28));

figure(2);
plot_and_keypress(time_vector(1:length(signal_in)),signal_in,channelNameArray(30));

figure(3);
plot_and_keypress(time_vector(1:length(signal_out2)),signal_out2,channelNameArray(30));

% 
% plot_and_keypress(time_vector(1:length(eegData)),eegDataT(1:30,:),channelNameArray(1:30));