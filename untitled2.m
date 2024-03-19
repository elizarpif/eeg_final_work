%selection of windows
EEG = load('EEG3.mat');
EEG3 = EEG.EEG';

fnyquist = 256.160664;
fs = fnyquist*2; 
ts = 1/fs;

%new sampling and nyquist frequencies
fnyquist_new = 256.160664/2;
fs_new = fnyquist_new*2; 


TBR05 = EEG3(51, 400/ts:405/ts);

% TBR05
figure(1); 
[p, f] = periodogram(TBR05);
plot(f * fs / (2*pi), 10*log10(p));
title('TBR05');
xlabel('Frequency (Hz)')
ylabel('Power/frequency (dB/(rad/sample))');
% hold on; 
% yline(0, '--r');
% xline(fnyquist, '--r');
hold off; 


TPL04 = EEG3(28, 484/ts:488/ts);
figure(2); 
[p, f] = periodogram(TPL04);
plot(f * fs / (2*pi), 10*log10(p));
title('TPL04');
xlabel('Frequency (Hz)')
ylabel('Power/frequency (dB/(rad/sample))');
% hold on; 
% yline(0, '--r');
% xline(fnyquist, '--r');
hold off; 