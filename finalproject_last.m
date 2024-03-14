EEG = load('EEG3.mat');
EEG3 = EEG.EEG';

fnyquist = 256.160664;
fs = fnyquist*2;
ts = 1/fs;

%new sampling and nyquist frequencies
fnyquist_new = 256.160664/2;
fs_new = fnyquist_new*2;

%selection of windows
TPL06 = EEG3(30, 26/ts:46/ts);
TPL04 = EEG3(28, 485/ts:505/ts);
TBR05 = EEG3(51, 400/ts:420/ts);
DER07 = EEG3(19, 362/ts:382/ts);
DEL01 = EEG3(1, 188/ts:208/ts);
TPR03 = EEG3(35, 262/ts:282/ts);

% %verifying aliasing
% figure();
% [p, f] = periodogram(TPL06);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TPL06');
% xlabel('Frequency (Hz)');
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(TPL04);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TPL04');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(TBR05);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TBR05');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(DER07);
% plot(f * fs / (2*pi), 10*log10(p));
% title('DER07');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(DEL01);
% plot(f * fs / (2*pi), 10*log10(p));
% title('DEL01');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(TPR03);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TPR03');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;

%butterworth filter
fcutoff = fnyquist_new*0.9;
order = 8;
[b, a] = butter(order, fcutoff/fnyquist, 'low');

% Aplication of filter
TPL06_filtered = filter(b, a, TPL06);
TPL04_filtered = filter(b, a, TPL04);
TBR05_filtered = filter(b, a, TBR05);
DER07_filtered = filter(b, a, DER07);
DEL01_filtered = filter(b, a, DEL01);
TPR03_filtered = filter(b, a, TPR03);

% %verification
% figure();
% [p, f] = periodogram(TPL06_filtered);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TPL06\_filtered');
% xlabel('Frequency (Hz)');
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(TPL04_filtered);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TPL04\_filtered');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(TBR05_filtered);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TBR05\_filtered');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(DER07_filtered);
% plot(f * fs / (2*pi), 10*log10(p));
% title('DER07\_filtered');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(DEL01_filtered);
% plot(f * fs / (2*pi), 10*log10(p));
% title('DEL01\_filtered');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;
%
% figure();
% [p, f] = periodogram(TPR03_filtered);
% plot(f * fs / (2*pi), 10*log10(p));
% title('TPR03\_filtered');
% xlabel('Frequency (Hz)')
% ylabel('Power/frequency (dB/(rad/sample))');
% hold on;
% yline(0, '--r');
% xline(fnyquist_new, '--r');
% hold off;

%downsampling
TPL06_down_filtered = TPL06_filtered(1:2:end);
TPL04_down_filtered = TPL04_filtered(1:2:end);
TBR05_down_filtered = TBR05_filtered(1:2:end);
DER07_down_filtered = DER07_filtered(1:2:end);
DEL01_down_filtered = DEL01_filtered(1:2:end);
TPR03_down_filtered = TPR03_filtered(1:2:end);

filtered_signals = [TPL06_down_filtered; TPL04_down_filtered; TBR05_down_filtered; DER07_down_filtered; DEL01_down_filtered; TPR03_down_filtered];

%%
% Calculo de surrogates para cada señal
for i = 1:6
    % Calcular surrogates para la señal actual
    for j = 2:20
        surrogate_index = (i - 1) * 20 + j; % Índice para la fila de los surrogates
        surrogates_matrix(surrogate_index, :) = phase_randomized_surrogate_aulaglobal(filtered_signals(i, :));
    end
    % Agregar la señal original a la matriz
    original_signal_index = (i - 1) * 20 + 1; % Índice para la fila de la señal original
    surrogates_matrix(original_signal_index, :) = filtered_signals(i, :);
end

%%
%nonlinear prediction error

embedding_dim = 6;
num_neighbors = 3;
theiler_correction = 30;
prediction_horizon = 2;
time_delays = 1:3:49;

for tau=1:3:49

    for signal_idx = 1:size(surrogates_matrix, 1)

        normalized_signal = normalize(surrogates_matrix(signal_idx,:));

        N = length(normalized_signal);

        ene = (embedding_dim - 1) * tau;

        % Compute delay matrix

        delay_matrix = delay_fuction(normalized_signal, embedding_dim, tau);

        % Compute distance matrix

        distance_matrix = distance_function(delay_matrix, embedding_dim, tau);

        % Plot the distances matrix
        % figure;
        % imagesc(distance_matrix);
        % title(['Distance Matrix of Signal ', num2str(signal_idx), ' with Delayed Signal by ', num2str(tau), ' Time [a.u.]']);
        % colorbar;

        % Apply corrections

        corrected_distance_matrix = corrected_matrix(distance_matrix,theiler_correction,prediction_horizon,embedding_dim, tau);

        %   Plot the corrected distances matrix
        % figure;
        % imagesc(corrected_distance_matrix);
        % title(['Distance Matrix of Signal ', num2str(signal_idx), ' with Delayed Signal by ', num2str(tau), ' Time [a.u.]']);
        % colorbar;

        epsilon_sum = 0;

        for i0 = (ene + 1 ): N - prediction_horizon

            reference = i0 ;

            [neighbours_distances,neighbours_indices] = compute_neighbours(reference, corrected_distance_matrix,num_neighbors);

            epsilon = compute_epsilon (i0,neighbours_distances,neighbours_indices, prediction_horizon, normalized_signal);

            epsilon_sum = epsilon_sum + epsilon;

        end

        prediction_error (signal_idx,tau) = compute_prediction_error(N,epsilon_sum, prediction_horizon, embedding_dim,tau);

    end
end


figure;
hold on;
%aprox trada 20 mins para cada ventana
% for i = 1:size(xx, 1) IF IT HAD MORE CHANNELS
plot(time_delays, prediction_error(1,time_delays), 'LineWidth', 3, 'Marker', 'd', 'Color', 'red');
plot(time_delays, prediction_error(2:20,time_delays), 'LineWidth', 1, 'Marker', 'd', 'Color', 'red', 'LineStyle','--');

plot(time_delays, prediction_error(21,time_delays), 'LineWidth', 2, 'Marker', 'd', 'Color', 'blue');
plot(time_delays, prediction_error(22:40,time_delays), 'LineWidth', 1, 'Marker', 'd', 'Color', 'blue', 'LineStyle','--');

plot(time_delays, prediction_error(41,time_delays), 'LineWidth', 2, 'Marker', 'd', 'Color', 'green');
plot(time_delays, prediction_error(42:60,time_delays), 'LineWidth', 1, 'Marker', 'd', 'Color', 'green', 'LineStyle','--');

plot(time_delays, prediction_error(61,time_delays), 'LineWidth', 2, 'Marker', 'd', 'Color', 'magenta');
plot(time_delays, prediction_error(62:80,time_delays), 'LineWidth', 1, 'Marker', 'd', 'Color', 'magenta', 'LineStyle','--');

plot(time_delays, prediction_error(81,time_delays), 'LineWidth', 2, 'Marker', 'd', 'Color', 'black');
plot(time_delays, prediction_error(82:100,time_delays), 'LineWidth', 1, 'Marker', 'd', 'Color', 'black', 'LineStyle','--');

plot(time_delays, prediction_error(101,time_delays), 'LineWidth', 2, 'Marker', 'd', 'Color', 'cyan');
plot(time_delays, prediction_error(102:120,time_delays), 'LineWidth', 1, 'Marker', 'd', 'Color', 'cyan', 'LineStyle','--');

hold off;

xlabel('Time Delay');
ylabel('Prediction Error');
title('Prediction Error for Each Signal');
legend('Normal signal', 'Surrogates normal signal', 'Channel thickening', 'Surrogates Channel Thickening', ...
    'Spaced peaks', 'Surrogates spaced peaks', 'Seizure beginning', 'Surrogates seizure beginning', 'After seizure', ...
    'Surrogates after seizure', 'Seizure', 'Surrogates seizure');

grid on; % Add grid


%%
% Function to compute delay vector

function delay_matrix = delay_fuction(signal, embedding_dim, delay)
N = length(signal);
ene = (embedding_dim - 1) * delay;

for i = ene + 1 : N
    for j = 1: embedding_dim
        delay_matrix(i,j) = signal(i - (j-1) * delay);
    end
end

end

%%

% Function to compute the distance matrix

function distance_matrix = distance_function(delay_matrix, embedding_dim, delay)

N = size(delay_matrix, 1);
ene = (embedding_dim - 1) * delay;
distance_matrix = zeros(N, N);

for i = (ene + 1) : N
    for j = (i + 1 ): N
        distance = sqrt(sum((delay_matrix(i, :) - delay_matrix(j, :)).^2));
        distance_matrix(i, j) = distance;
        distance_matrix(j, i) = distance;
    end
end

end

%%

% Fuction to apply Theiler's correction
function [corrected_distance_matrix] = corrected_matrix(distance_matrix,theiler_correction,horizon, embedding_dim, delay)

N = length (distance_matrix);
ene = (embedding_dim - 1) * delay;
corrected_distance_matrix = nan(N);

% Iterate over rows of the distance matrix
for i = (ene + 1) : N - horizon

    % Go through the upper right corner applying the correction
    for j = (i + theiler_correction) : N - horizon

        corrected_distance_matrix (i,j) = distance_matrix(i,j);
        corrected_distance_matrix (j,i) = distance_matrix(i,j);

    end
end

end

%%
% Fuction to get the neighbours
function [neighbours_distances,neighbours_indices] = compute_neighbours(reference, corrected_distance_matrix,k)

% Compute the distance matrix and sort it
distance_vector_reference = corrected_distance_matrix(reference, :);
[sorted_distances, sorted_indices] = sort(distance_vector_reference);

% Extract the k nearest neighbors
neighbours_distances = sorted_distances(1:k);
neighbours_indices = sorted_indices(1:k);

end

%%
% Fuction for computing epsilon
function epsilon = compute_epsilon (ti0 ,neighbours_distances, neighbours_indices, horizon,signal)
% Find indices of non-NaN distances
valid_indices = ~isnan(neighbours_distances);

% Extract valid neighbours
valid_neighbours_indices = neighbours_indices(valid_indices);

% Compute epsilon using valid neighbours
sum_neig = 0;
k = length(valid_neighbours_indices);
for s = 1 : k
    tis = neighbours_indices(s);
    sum_neig = sum_neig + signal(tis+ horizon); %sum of the errors of the neighbours
end

mean_neig = sum_neig / k ;

epsilon = (signal(ti0 + horizon) - mean_neig)^2;

end


%%
% Fuction to compute the prediction error

function prediction_error = compute_prediction_error(N, epsilon_sum, horizon, embedding_dim, delay)

ene = (embedding_dim - 1) * delay;

prediction_error = sqrt(1 / (N - horizon - ene) * sum(epsilon_sum));

end


