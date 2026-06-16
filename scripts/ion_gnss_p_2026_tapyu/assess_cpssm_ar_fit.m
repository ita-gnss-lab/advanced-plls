clearvars; clc;

addpath(genpath(fullfile(pwd, "..", "..", "..")));

sim_time = 150;
t_samp = 0.01;
fs = 1/t_samp;

seed = 8;
rng(seed);

severities = {'Weak','Strong'};
cpssm_params = struct( ...
    'Weak',    {'weak', 'is_enable_cmd_print', false, 'simulation_time', sim_time, 'sampling_interval', t_samp, 'rhof_veff_ratio', 1.5},...
    'Strong',  {'strong', 'is_enable_cmd_print', false, 'simulation_time', sim_time, 'sampling_interval', t_samp, 'rhof_veff_ratio', 0.27}...
    );

[psi_weak_ts, phi_R_weak_ts] = get_tppsm_multifreq_data(cpssm_params.Weak, 'seed', seed);
[psi_strong_ts, phi_R_strong_ts] = get_tppsm_multifreq_data(cpssm_params.Strong, 'seed', seed);

wrapped_phi_I_weak_ts = angle(psi_weak_ts(:,1));
wrapped_phi_I_strong_ts = angle(psi_strong_ts(:,1));

phases_ts = [wrapped_phi_I_weak_ts, wrapped_phi_I_strong_ts];

ar_orders_amount = 20;
bic_array = zeros(ar_orders_amount,2);
% Estimate `n_orders` AR models for assessing the optimum model order
% First for loop -> Weak and Strong 
for severity = 1:2
    for ar_order = 1:ar_orders_amount
        ar_model = arima(ar_order,0,0);
        ar_model.Constant = 0;
        ar_model = estimate(ar_model, phases_ts(:, severity));
        results = summarize(ar_model);
        bic_array(ar_order,severity) = results.BIC;
    end
end

[~, min_idxs]= min(bic_array);

% Estimate optimal models
opt_ar_model_weak = arima(min_idxs(1),0,0);
opt_ar_model_weak.Constant = 0;
opt_ar_model_weak = estimate(opt_ar_model_weak, phases_ts(:, 1));

opt_ar_model_strong = arima(min_idxs(2),0,0);
opt_ar_model_strong.Constant = 0;
opt_ar_model_strong = estimate(opt_ar_model_strong, phases_ts(:, 2));

% Get residuals
res_weak = infer(opt_ar_model_weak, phases_ts(:,1));
res_strong = infer(opt_ar_model_strong, phases_ts(:,2));

acf_weak = autocorr(res_weak);
acf_strong = autocorr(res_strong);

win = hamming(length(phases_ts(:,1)));
nfft = 2^16;
[pxx_weak,~] = periodogram(phases_ts(:,1),win,nfft,"onesided","psd",fs);
[pxx_strong,f] = periodogram(phases_ts(:,2),win,nfft,"onesided","psd",fs);

S_weak = compute_ar_psd(cell2mat(opt_ar_model_weak.AR), t_samp, opt_ar_model_weak.Variance, f);
S_strong = compute_ar_psd(cell2mat(opt_ar_model_strong.AR), t_samp, opt_ar_model_strong.Variance, f);

%% Save results

data.frequency_support = f;
data.pxx_weak = pxx_weak;
data.pxx_strong = pxx_strong;
data.S_AR_weak = S_weak;
data.S_AR_strong = S_strong;
data.bic_array_weak = bic_array(:,1);
data.bic_array_strong = bic_array(:,2);
data.opt_ar_order_weak = min_idxs(1);
data.opt_ar_order_strong = min_idxs(2);

save('results/cpssm_ar_fit.mat', 'data');


%% Figures
figure('Color','w', 'Position', [50,50,1200,550]);
tiledlayout(2,2,'TileSpacing','compact','Padding','tight');
font_size = 18;
% BIC vs AR order
nexttile;
% plot strong then weak
h_strong = plot(1:ar_orders_amount, bic_array(:,2), 's--', 'LineWidth',1.5); hold on;
h_weak   = plot(1:ar_orders_amount, bic_array(:,1), 'o-' , 'LineWidth',1.5);

% plot stars at the minimal‐BIC orders
h_opt_strong = plot(min_idxs(2), bic_array(min_idxs(2),2), '*' , ...
    'MarkerSize',12, 'LineWidth',1.5, 'Color', h_strong.Color);
h_opt_weak   = plot(min_idxs(1), bic_array(min_idxs(1),1), '*' , ...
    'MarkerSize',12, 'LineWidth',1.5, 'Color', h_weak.Color);

xlabel('AR order');
ylabel('BIC');
title('BIC vs AR order');
grid on;
legend([h_strong, h_weak, h_opt_strong, h_opt_weak], ...
       {'Strong','Weak','Optimal order Strong','Optimal order Weak'}, ...
       'Location','east');
set(gca,'FontName','Times New Roman', 'FontSize', font_size);

% Residual time-series
nexttile;
t = (0:length(res_strong)-1) * t_samp;
plot(t, res_strong, '--','LineWidth',1); hold on;   % strong
plot(t, res_weak,   '-','LineWidth',1);            % weak
xlabel('Time [s]'); ylabel('Residual [rad]');
legend('Strong','Weak','Location','best');
title('Residuals');
grid on;
set(gca,'FontName','Times New Roman', 'FontSize', font_size);

% Residuals ACF
nexttile;
lags = 0:numel(acf_strong)-1;
stem(lags, acf_strong,   'filled'); hold on;   % strong
stem(lags, acf_weak,     'filled');           % weak
xlabel('Lag'); ylabel('Normalized ACF [rad^2]');
legend('Strong','Weak','Location','best');
title('Residuals ACF');
grid on;
set(gca,'FontName','Times New Roman', 'FontSize', font_size);

% Periodogram vs AR PSD
nexttile;
plot(f, 10*log10(pxx_strong),   '--','LineWidth',1); hold on;   % strong periodogram
plot(f, 10*log10(S_strong),     '-.','LineWidth',1.5);         % strong AR‐PSD
plot(f, 10*log10(pxx_weak),     '-','LineWidth',1);            % weak periodogram
plot(f, 10*log10(S_weak),       ':','LineWidth',1.5);          % weak AR‐PSD
xlabel('Frequency [Hz]'); ylabel('PSD [dB-Hz]');
legend('Periodogram Strong','AR-PSD Strong','Periodogram Weak','AR-PSD Weak','Location','best');
title('Periodogram vs AR PSD');
set(gca, 'XScale', 'log');
grid on;
set(gca,'FontName','Times New Roman', 'FontSize', font_size);

% make results folder
if ~exist("results","dir")
    mkdir("results");
end

exportgraphics(gcf, 'results/cpssm_ar_fit.pdf', 'ContentType','vector');

function S_gg = compute_ar_psd(ar_coeffs, t_samp, sigma2_ar, freq)
% COMPUTE_AR_PSD  Compute S_{γγ}(f) for an AR(p) process
%
%   ar_coeffs  = [a1, a2, …, a_p]         (1×p vector of AR coefficients)
%   ti         = T_I                       (scalar integration interval)
%   sigma2_ar  = σ_AR^2                    (scalar AR‐process variance)
%   freq       = vector of frequencies    (Hz)
%
% Returns:
%   S_gg       = PSD values at each freq  (same size as freq)

    % ensure column vector for freq
    freq = freq(:);
    p    = numel(ar_coeffs);

    % build matrix of e^{-j2π f T_I i} terms, size = [numel(freq) × p]
    i_matrix = 1:p;  % 1×p
    E = exp(-1j*2*pi * bsxfun(@times, freq*t_samp, i_matrix));  

    % form the denominator H(f) = 1 – sum_i a_i e^{-j2π f T_I i}
    H = 1 - E * ar_coeffs(:);

    % PSD: 2·T_I·σ_AR^2 · |1 / H(f)|^2
    S_gg = 2 * t_samp * sigma2_ar * abs(1 ./ H).^2;
end