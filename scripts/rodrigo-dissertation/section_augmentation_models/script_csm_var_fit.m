% script_csm_ar_goodness_of_fit.m
%
% Script to identify the goodness of fit of the AR model for scintillation 
% amplitude and phase time series generated by the Cornell scintillation
% model (CSM) using the ARFIT algorithm [1].
%
% Steps:
%   1) Monte Carlo assessment of optimal AR orders per severity.
%   2) Extraction of residuals for the most frequent optimal orders.
%   3) Calculation of one-sided ACFs of those residuals.
%   4) Comparison of periodograms of the synthetic signals generated using 
%      the CSM vs. estimated AR model's power spectral densities.
%
% References:
% [1] Schneider, Tapio, and Arnold Neumaier. “Algorithm 808: ARfit—a Matlab 
% Package for the Estimation of Parameters and Eigenmodes of Multiariate
% Autoregressive Models.” ACM Trans. Math. Softw. 27, no. 1 
% (March 1, 2001): 58–65. https://doi.org/10.1145/382043.382316.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

clearvars; clc;
addpath(genpath(fullfile(pwd,'..','..','..', 'libs')));

% Setup output folders
fig_dir = 'pdf_figures_csm'; % where we’ll write vector PDFs
csv_dir = 'csv_data_csm';    % where we’ll write CSV tables

if ~exist(fig_dir,'dir')
    mkdir(fig_dir);
end
if ~exist(csv_dir,'dir')
    mkdir(csv_dir);
end

%% Simulation parameters
simulation_time = 300;
sampling_interval = 0.01;
severities = {'Weak','Moderate','Strong'};
csm_params = struct( ...
    'Weak',    struct('S4', 0.2, 'tau0', 1, 'simulation_time', simulation_time, 'sampling_interval', sampling_interval),...
    'Moderate',struct('S4', 0.5, 'tau0', 0.6, 'simulation_time', simulation_time, 'sampling_interval', sampling_interval),...
    'Strong',  struct('S4', 0.9, 'tau0', 0.2, 'simulation_time', simulation_time, 'sampling_interval', sampling_interval)...
);
font_size = 11;

%% Monte Carlo optimal AR model order assessment
mc_runs    = 300;
min_order  = 1;
max_order  = 30;
optimal_orders_amp   = zeros(mc_runs,numel(severities));
optimal_orders_phase = zeros(mc_runs,numel(severities));
sbc_amp_array = zeros(mc_runs, numel(severities), max_order - min_order + 1);
sbc_phase_array = zeros(mc_runs, numel(severities), max_order - min_order + 1);

seed = 1;
for mc_idx = 1:mc_runs
    for i = 1:numel(severities)
        severity = severities{i};
        rng(seed);
        data      = get_csm_data(csm_params.(severity));
        amp_ts    = abs(data);
        phase_ts  = atan2(imag(data),real(data));

        [~, A_amp, ~, sbc_amp]   = arfit(amp_ts,   min_order, max_order);
        [~, A_phase, ~, sbc_phase] = arfit(phase_ts, min_order, max_order);

        optimal_orders_amp(mc_idx,i)   = size(A_amp,2)/size(A_amp,1);
        optimal_orders_phase(mc_idx,i) = size(A_phase,2)/size(A_phase,1);
        
        sbc_amp_array(mc_idx, i, :) = sbc_amp;
        sbc_phase_array(mc_idx, i, :) = sbc_phase;

        seed = seed + 1;
    end
end

orders       = min_order:max_order;
counts_amp   = zeros(numel(orders),numel(severities));
counts_phase = zeros(numel(orders),numel(severities));
for i = 1:numel(severities)
    counts_amp(:,i)   = histcounts(optimal_orders_amp(:,i),   [orders, orders(end)+1]);
    counts_phase(:,i) = histcounts(optimal_orders_phase(:,i), [orders, orders(end)+1]);
end

% Normalize counts to percentages
pct_amp   = counts_amp   / mc_runs * 100;
pct_phase = counts_phase / mc_runs * 100;

figure('Position',[100,100,1000,300]);
colors = lines(numel(severities));  % or get(gca,'ColorOrder')

% Amplitude
subplot(1,2,1);
hold on;
markers = {'o', 's', '^'};
for j = 1:numel(severities)
    plot(orders, pct_amp(:,j), ['-',markers{j}], ...
         'LineWidth',1.5, 'MarkerSize',6, 'Color',colors(j,:), ...
         'DisplayName',severities{j});
end
hold off;
xlabel('AR Model Order');
ylabel('Percentage of runs [%]');
title('Amplitude');
legend('Location','best');
set(gca, 'FontSize', font_size);
grid on;

% Phase
subplot(1,2,2);
hold on;
for j = 1:numel(severities)
    plot(orders, pct_phase(:,j), ['-',markers{j}], ...
         'LineWidth',1.5, 'MarkerSize',6, 'Color',colors(j,:), ...
         'DisplayName',severities{j});
end
hold off;
xlabel('AR Model Order');
ylabel('Percentage of runs [%]');
title('Phase');
%legend('Location','best');
set(gca, 'FontSize', font_size);
grid on;

% Export Optimal AR Order plot & CSV
fig_name = 'optimal_ar_order_frequency_csm';
exportgraphics(gcf, fullfile(fig_dir,[fig_name,'.pdf']), 'ContentType','vector');
T_opt = table( ...
  orders(:), ...
  pct_amp(:,1), pct_amp(:,2), pct_amp(:,3), ...
  pct_phase(:,1), pct_phase(:,2), pct_phase(:,3), ...
  'VariableNames',{ ...
    'Order', ...
    'PctAmp_Weak','PctAmp_Moderate','PctAmp_Strong', ...
    'PctPhs_Weak','PctPhs_Moderate','PctPhs_Strong' } );
writetable(T_opt, fullfile(csv_dir,[fig_name,'.csv']));

% Plot the mean SBC ------------------------------------------------------

mean_sbc_amp_array = squeeze(mean(sbc_amp_array,1)).';
mean_sbc_phase_array = squeeze(mean(sbc_phase_array,1)).';

figure('Position',[100,100,1000,250]);

% Amplitude subplot
subplot(1,2,1);
colors = get(gca,'ColorOrder');
plot(orders, mean_sbc_amp_array, 'LineWidth', 1.5);
hold on;
for j = 1:size(mean_sbc_amp_array,2)
    [minVal, minIdx] = min(mean_sbc_amp_array(:,j));
    plot(orders(minIdx), minVal, '*', ...
         'MarkerSize',10, 'Color', colors(j,:));
end
xlabel('AR Model Order');
ylabel('Mean SBC');
title('Amplitude');
legend(severities,'Location','northeast');
set(gca, 'FontSize', font_size);
grid on;
hold off;

% Phase subplot (now using plot instead of bar)
subplot(1,2,2);
colors = get(gca,'ColorOrder');
plot(orders, mean_sbc_phase_array, 'LineWidth', 1.5);
hold on;
for j = 1:size(mean_sbc_phase_array,2)
    [minVal, minIdx] = min(mean_sbc_phase_array(:,j));
    plot(orders(minIdx), minVal, '*', ...
         'MarkerSize',10, 'Color', colors(j,:));
end
xlabel('AR Model Order');
ylabel('Mean SBC');
title('Phase');
%legend(severities,'Location','best');
set(gca, 'FontSize', font_size);
grid on;
hold off;

% Export Mean SBC plot & CSV
fig_name = 'mean_sbc_csm';
exportgraphics(gcf, fullfile(fig_dir,[fig_name,'.pdf']), 'ContentType','vector');
T_sbc = table( ...
  orders(:), ...
  mean_sbc_amp_array(:,1), mean_sbc_amp_array(:,2), mean_sbc_amp_array(:,3), ...
  mean_sbc_phase_array(:,1), mean_sbc_phase_array(:,2), mean_sbc_phase_array(:,3), ...
  'VariableNames',{ ...
    'Order', ...
    'MeanSBC_Amp_Weak','MeanSBC_Amp_Moderate','MeanSBC_Amp_Strong', ...
    'MeanSBC_Phs_Weak','MeanSBC_Phs_Moderate','MeanSBC_Phs_Strong' } );
writetable(T_sbc, fullfile(csv_dir,[fig_name,'.csv']));


%% Obtain residuals for most frequent orders
[~, min_sbc_idx_amp]   = min(mean_sbc_amp_array,[],1);
[~, min_sbc_idx_phase] = min(mean_sbc_phase_array,[],1);

residuals = struct('amplitude',[],'phase',[]);
for i = 1:numel(severities)
    severity   = severities{i};
    rng(i);
    data       = get_csm_data(csm_params.(severity));
    amp_ts     = abs(data);
    phase_ts   = atan2(imag(data),real(data));

    ord_amp    = orders(min_sbc_idx_amp(i));
    ord_phase  = orders(min_sbc_idx_phase(i));

    [w_amp,   A_amp]   = arfit(amp_ts,   ord_amp,   ord_amp);
    [w_phase, A_phase] = arfit(phase_ts, ord_phase, ord_phase);

    [~, res_amp]   = arres(w_amp,   A_amp,   amp_ts,   20);
    [~, res_phase] = arres(w_phase, A_phase, phase_ts, 20);

    residuals.amplitude.(severity) = [NaN(ord_amp,1);   res_amp];
    residuals.phase.(severity)     = [NaN(ord_phase,1); res_phase];
end

% Plot residuals in order: Strong, Moderate, Weak, vs. time
% Invert colors: strong→yellow, moderate→redish, weak→blueish
plot_order = {'Strong','Moderate','Weak'};
time = sampling_interval : sampling_interval : simulation_time;

% Grab default line‐colors and reverse their order
base_colors = lines(numel(plot_order));
colors      = base_colors([3,2,1],:);  

figure('Position',[100,100,1000,250]);

% Amplitude residuals
subplot(1,2,1); hold on;
for k = 1:numel(plot_order)
    sev = plot_order{k};
    plot(time, residuals.amplitude.(sev), ...
         'LineWidth',1, 'Color',colors(k,:), ...
         'DisplayName', sev);
end
hold off;
xlabel('Time [s]');
ylabel('Residuals');
title('Amplitude Residuals');
legend('Location','best', 'Direction','reverse');
set(gca, 'FontSize', font_size);
grid on;

% Phase residuals
subplot(1,2,2); hold on;
for k = 1:numel(plot_order)
    sev = plot_order{k};
    plot(time, residuals.phase.(sev), ...
         'LineWidth',1, 'Color',colors(k,:), ...
         'DisplayName', sev);
end
hold off;
xlabel('Time [s]');
ylabel('Residuals [rad]');
title('Phase Residuals'); 
%legend('Location','best', 'Direction','reverse');
set(gca, 'FontSize', font_size);
grid on;

% Export residuals plot & CSV
fig_name = 'residuals_csm';
exportgraphics(gcf, fullfile(fig_dir,[fig_name,'.pdf']), 'ContentType','vector');
T_res = table( ...
  time(:), ...
  residuals.amplitude.Weak,   residuals.amplitude.Moderate,   residuals.amplitude.Strong, ...
  residuals.phase.Weak,       residuals.phase.Moderate,       residuals.phase.Strong, ...
  'VariableNames',{ ...
    'Time_s', ...
    'ResAmp_Weak','ResAmp_Moderate','ResAmp_Strong', ...
    'ResPhs_Weak','ResPhs_Moderate','ResPhs_Strong' } );
writetable(T_res, fullfile(csv_dir,[fig_name,'.csv']));

% Compute one-sided ACFs of residuals -------------------------------------

% Amount of lags on the ACF
lags_amount = 20;
stem_width  = 1.5;
markers     = struct('Weak','o','Moderate','s','Strong','^');
acfs        = struct('amplitude',[],'phase',[]);

for i = 1:numel(severities)
    severity = severities{i};
    
    % Removing the NaNs of the amplitude and phase residuals
    amp_res  = residuals.amplitude.(severity);
    amp_res  = amp_res(~isnan(amp_res));
    phs_res  = residuals.phase.(severity);
    phs_res  = phs_res(~isnan(phs_res));

    acf_amp          = xcorr(amp_res, amp_res, lags_amount, 'normalized');
    acf_phs          = xcorr(phs_res, phs_res, lags_amount, 'normalized');
    acfs.amplitude.(severity) = acf_amp(lags_amount+1:end);
    acfs.phase.(severity)     = acf_phs(lags_amount+1:end);
end

time_lag = (0:lags_amount) * sampling_interval;
figure('Position',[100,100,1000,250]);
subplot(1,2,1);
hold on;
for i = 1:numel(severities)
    severity = severities{i};
    stem(time_lag, acfs.amplitude.(severity), 'LineWidth', stem_width, ...
         'Marker', markers.(severity), 'DisplayName', severity);
end
hold off;
xlabel('Time Lag [s]'); ylabel('Normalized ACF');
title('Amplitude Residuals ACF');
legend('Location','best'); grid on;
set(gca, 'FontSize', font_size);

subplot(1,2,2);
hold on;
for i = 1:numel(severities)
    severity = severities{i};
    stem(time_lag, acfs.phase.(severity), 'LineWidth', stem_width, ...
         'Marker', markers.(severity), 'DisplayName', severity);
end
hold off;
xlabel('Time Lag [s]'); ylabel('Normalized ACF [rad^2]');
title('Phase Residuals ACF');
set(gca, 'FontSize', font_size);
%legend('Location','best'); 
grid on;

% Export residuals ACF plot & CSV
fig_name = 'residuals_acf_csm';
exportgraphics(gcf, fullfile(fig_dir,[fig_name,'.pdf']), 'ContentType','vector');
T_acf = table( ...
  time_lag(:), ...
  acfs.amplitude.Weak,   acfs.amplitude.Moderate,   acfs.amplitude.Strong, ...
  acfs.phase.Weak,       acfs.phase.Moderate,       acfs.phase.Strong, ...
  'VariableNames',{ ...
    'Lag_s', ...
    'ACF_Amp_Weak','ACF_Amp_Moderate','ACF_Amp_Strong', ...
    'ACF_Phs_Weak','ACF_Phs_Moderate','ACF_Phs_Strong' } );
writetable(T_acf, fullfile(csv_dir,[fig_name,'.csv']));


%% Monte Carlo Periodogram vs. AR PSD comparison

% Number of points on the frequency domain
nfft = 2^16;
% Sampling frequency in Hz
fs   = 1/sampling_interval;
% Number of Monte Carlo realizations
num_realizations = 300;
% Amount of samples in the time series
N        = simulation_time * fs;
% Windowing function --- Hamming window
win      = hamming(N); 
% Amount of overlapping samples
noverlap = 0; 

% Struct for pre-allocating the frequency support, the periodograms and 
% the AR model's.
psd_comparison = struct( ...
  'freq',     [], ...
  'amplitude', struct('periodogram',[],'ar_psd',[]), ...
  'phase',     struct('periodogram',[],'ar_psd',[]) ...
);

for i = 1:numel(severities)
    sev = severities{i};

    %---- initialize accumulators ----------------------------------------
    acc_periodogram_amp = zeros(nfft/2+1,1);
    acc_periodogram_phs = zeros(nfft/2+1,1);
    acc_ar_amp         = zeros(nfft/2+1,1);
    acc_ar_phs         = zeros(nfft/2+1,1);

    for mc = 1:num_realizations
        rng(i + mc);  % ensure reproducibility across severities and MC

        %---- 1) Generate & center signals --------------------------------
        x        = get_csm_data(csm_params.(sev));
        amp_ts   = abs(x);
        phase_ts = atan2(imag(x), real(x));

        ctr_amp_ts = amp_ts - mean(amp_ts);
        ctr_phs_ts = phase_ts - mean(phase_ts);
        % 
        % % 1) pick a window of the same length as your (centered) time‐series
        % N     = numel(ctr_amp_ts);
        % win   = hamming(N);           % or any other window: blackman, hann, etc.
        % U     = sum(win.^2);          % window energy → normalization constant
        % 
        % % 2) window the data, zero‐pad to nfft, take FFT
        % Xw_amp = fft( ctr_amp_ts .* win, nfft );
        % Xw_phs = fft( ctr_phs_ts .* win, nfft );
        % 
        % % 3) form one‐sided PSD, normalize by fs*U and double for positive freqs
        % S_amp_full = 2 * ( Xw_amp .* conj(Xw_amp) ) / (fs * U);
        % S_phs_full = 2 * ( Xw_phs .* conj(Xw_phs) ) / (fs * U);
        % 
        % % 4) keep only the first half
        % half = 1:(nfft/2+1);
        % periodogram_amp = real( S_amp_full(half) );
        % periodogram_phs = real( S_phs_full(half) );
        
        % 2) use cpsd to get your one-sided, Hamming-tapered PSD estimate
        [per_amp, f] = cpsd( ...
            ctr_amp_ts, ctr_amp_ts, ...   % x and y both = your signal  
            win, noverlap, nfft, fs ...   % window, overlap, fft‐length, fs
        );
        [per_phs, ~] = cpsd( ...
            ctr_phs_ts, ctr_phs_ts, ...
            win, noverlap, nfft, fs ...
        );

        % 3) grab the frequency axis (it already goes from 0 to fs/2)
        if isempty(psd_comparison.freq)
            psd_comparison.freq = f;
        end

        %---- 4) Fit AR(p) & compute AR-PSD -----------------------------
        p_amp = orders(min_sbc_idx_amp(i));
        p_phs = orders(min_sbc_idx_phase(i));
        [w_amp, A_amp, C_amp] = arfit(amp_ts,   p_amp, p_amp);
        [w_phs, A_phs, C_phs] = arfit(phase_ts, p_phs, p_phs);

        a_amp = [1, -reshape(A_amp,1,[])];
        a_phs = [1, -reshape(A_phs,1,[])];

        z = exp(-1j*2*pi*psd_comparison.freq*sampling_interval);

        H_amp        = 1 ./ sum(a_amp .* z.^(-(0:p_amp)), 2);
        H_phs        = 1 ./ sum(a_phs .* z.^(-(0:p_phs)), 2);

        % One-sided PSD of the AR model
        % NOTE: The factor 2 raises from the fact the we're computing the
        % one-sided PSD.
        S_ar_amp = 2 * (C_amp/fs) * abs(H_amp).^2;
        S_ar_phs = 2 * (C_phs/fs) * abs(H_phs).^2;

        %---- accumulate results ------------------------------------------
        acc_periodogram_amp = acc_periodogram_amp + per_amp;
        acc_periodogram_phs = acc_periodogram_phs + per_phs;
        acc_ar_amp         = acc_ar_amp         + S_ar_amp;
        acc_ar_phs         = acc_ar_phs         + S_ar_phs;
    end

    %---- 5) Compute Monte Carlo mean ------------------------------------
    mean_periodogram_amp = acc_periodogram_amp / num_realizations;
    mean_periodogram_phs = acc_periodogram_phs / num_realizations;
    mean_ar_amp         = acc_ar_amp         / num_realizations;
    mean_ar_phs         = acc_ar_phs         / num_realizations;

    %---- 6) Stash averaged results -------------------------------------
    psd_comparison.amplitude.(sev).periodogram = mean_periodogram_amp;
    psd_comparison.amplitude.(sev).ar_psd      = mean_ar_amp;
    psd_comparison.phase.(sev).periodogram     = mean_periodogram_phs;
    psd_comparison.phase.(sev).ar_psd          = mean_ar_phs;
end

%% Plot periodograms and AR PSDs
cmap_p = winter(numel(severities));  % periodogram lines
cmap_v = autumn(numel(severities));  % AR-PSD     lines

figure('Position',[100,100,1000,350]);
F = psd_comparison.freq;

% Amplitude
subplot(1,2,1); hold on;
h_amp = gobjects(2*numel(severities),1);
for k=1:numel(severities)
  sev = severities{k};
  idx = 2*(k-1)+1;
  h_amp(idx)   = plot(F, 10*log10(psd_comparison.amplitude.(sev).periodogram), '--', ...
                      'Color',cmap_p(k,:), 'LineWidth',1, 'DisplayName',[sev ' – Periodogram']);
  h_amp(idx+1) = plot(F, 10*log10(psd_comparison.amplitude.(sev).ar_psd),           '-', ...
                      'Color',cmap_v(k,:), 'LineWidth',2, 'DisplayName',[sev ' – AR PSD']);
end
hold off;
xlabel('Norm. freq. (× 1/T_I) [Hz]'); 
ylabel('PSD [dB/Hz]');
title('Amplitude');
legend(h_amp,'Location','best');
set(gca,'XScale','log','XLim',[1e-4*fs,0.4*fs], 'FontSize', font_size);
grid on; 


% Phase
subplot(1,2,2); hold on;
h_phs = gobjects(2*numel(severities),1);
for k=1:numel(severities)
  sev = severities{k};
  idx = 2*(k-1)+1;
  h_phs(idx)   = plot(F, 10*log10(psd_comparison.phase.(sev).periodogram), '--', ...
                      'Color',cmap_p(k,:), 'LineWidth',1, 'DisplayName',[sev ' – Periodogram']);
  h_phs(idx+1) = plot(F, 10*log10(psd_comparison.phase.(sev).ar_psd),           '-', ...
                      'Color',cmap_v(k,:), 'LineWidth',2, 'DisplayName',[sev ' – AR PSD']);
end
hold off;
xlabel('Norm. freq. (× 1/T_I) [Hz]');
ylabel('PSD [dB (rad^2/Hz)]');
title('Phase');
set(gca,'XScale','log','XLim',[1e-4*fs,0.4*fs], 'FontSize', font_size);
grid on; 

% Export PSD vs. AR PSD plot & CSV
fig_name = 'periodogram_vs_ar_psd_csm';
exportgraphics(gcf, fullfile(fig_dir,[fig_name,'.pdf']), 'ContentType','vector');
% make sure F = psd_comparison.freq is in workspace
T_psd = table( ...
  F(:), ...
  psd_comparison.amplitude.Weak.periodogram,   psd_comparison.amplitude.Moderate.periodogram,   psd_comparison.amplitude.Strong.periodogram, ...
  psd_comparison.amplitude.Weak.ar_psd,        psd_comparison.amplitude.Moderate.ar_psd,        psd_comparison.amplitude.Strong.ar_psd, ...
  psd_comparison.phase.Weak.periodogram,       psd_comparison.phase.Moderate.periodogram,       psd_comparison.phase.Strong.periodogram,  ...
  psd_comparison.phase.Weak.ar_psd,            psd_comparison.phase.Moderate.ar_psd,            psd_comparison.phase.Strong.ar_psd, ...
  'VariableNames',{ ...
    'Freq_Hz', ...
    'AmpPer_Weak','AmpPer_Moderate','AmpPer_Strong', ...
    'AmpAR_Weak','AmpAR_Moderate','AmpAR_Strong', ...
    'PhsPer_Weak','PhsPer_Moderate','PhsPer_Strong', ...
    'PhsAR_Weak','PhsAR_Moderate','PhsAR_Strong' } );
writetable(T_psd, fullfile(csv_dir,[fig_name,'.csv']));

