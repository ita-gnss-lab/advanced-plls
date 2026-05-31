% error_series_kf_ar_vs_sigma2_W.m
%
% Plot KF-AR phase error time series with and without geometry aid.
% Keeps only the KF-AR model and CPSSM scenarios.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

clearvars; clc;
addpath(genpath(fullfile(pwd,'..','..','..','libs')));

%% 1) Reproducibility & Config
seed              = 6;
rng(seed);
sampling_interval = 1e-2;
settling_time     = 50;
simulation_time   = 300;
cache_dir         = fullfile(fileparts(mfilename('fullpath')),'cache');

%% 2) Zoom window
zoom_start = 50;  % seconds
zoom_end   = 150; % seconds
idx_full   = round(zoom_start/sampling_interval):round(zoom_end/sampling_interval);
time_zoom  = (zoom_start:sampling_interval:zoom_end).';

%% 3) KF-AR config
[kf_ar_cfg, online_mdl_learning_cfg] = get_adaptive_cfgs();

%% 4) Plotting config
severities = {'weak', 'strong'};
model_list = {'cpssm_wo_refr', 'cpssm_w_refr'};
geometry_labels = {'KF-AR without geometry aid', 'KF-AR with geometry aid'};
geometry_styles = {'-', '--'};
geometry_colors = [0.10 0.40 0.80; 0.85 0.30 0.15];
font_size = 14;

%% 5) Common settings
sigma2_W_3 = 1e-6;
geometry_noise_variance = 1e-12;

%% 6) Loop through each CPSSM scenario
for m = 1:numel(model_list)
    mdl = model_list{m};
    switch mdl
        case 'cpssm_wo_refr'
            scint_cfg      = 'cpssm';
            training_scint = 'TPPSM';
            is_refractive_effects_removed_received_signal = true;
        case 'cpssm_w_refr'
            scint_cfg      = 'cpssm';
            training_scint = 'TPPSM';
            is_refractive_effects_removed_received_signal = false;
    end

    fig = figure('Units','normalized','Position',[0.05,0.05,0.9,0.9]); % ,'Color','w'
    tiledlayout(numel(severities),1,'TileSpacing','compact','Padding','compact');

    for row = 1:numel(severities)
        ax = nexttile; hold(ax,'on');
        sev = severities{row};

        [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, ar_phase_idx] = ...
            get_overall_cfgs(cache_dir, scint_cfg, sev, is_refractive_effects_removed_received_signal, sigma2_W_3, ...
            sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance);

        [rx_sig, true_los_phase, psi_settled, diffractive_phase] = get_received_signal(rx_signal_model_inputs{:});
        geometry_los_phase = get_noisy_geometry_phase(true_los_phase, geometry_noise_variance);

        [kf_ar_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', training_scint, kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
        [kf_ar_no_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', training_scint, kf_ar_cfg, online_mdl_learning_cfg);

        valid_los_phase = true_los_phase(idx_full, 1);
        if is_refractive_effects_removed_received_signal
            valid_cpssm_total_phase = diffractive_phase(idx_full, 1);
        else
            valid_cpssm_total_phase = get_corrected_phase(psi_settled(idx_full, 1));
        end
        valid_total_phase = valid_los_phase + valid_cpssm_total_phase;

        kf_ar_geo_total_phase = wrapToPi(kf_ar_geo(:,1) + kf_ar_geo(:,ar_phase_idx));
        kf_ar_no_geo_total_phase = wrapToPi(kf_ar_no_geo(:,1) + kf_ar_no_geo(:,ar_phase_idx));

        plot(ax, time_zoom, wrapToPi(kf_ar_no_geo_total_phase(idx_full) - valid_total_phase), ...
            'LineWidth', 1.5, 'Color', geometry_colors(1,:), 'LineStyle', geometry_styles{1}, ...
            'DisplayName', geometry_labels{1});
        plot(ax, time_zoom, wrapToPi(kf_ar_geo_total_phase(idx_full) - valid_total_phase), ...
            'LineWidth', 1.5, 'Color', geometry_colors(2,:), 'LineStyle', geometry_styles{2}, ...
            'DisplayName', geometry_labels{2});
        plot(ax, time_zoom, zeros(size(idx_full)), 'k', 'LineWidth', 1.2, 'HandleVisibility', 'off');

        grid(ax, 'on');
        grid(ax, 'minor');
        set(ax, 'FontName', 'Times New Roman', 'TickLabelInterpreter', 'latex', 'FontSize', font_size);
        ylabel(ax, sprintf('%s error [rad]', upper(sev)), 'Interpreter', 'none', 'FontName', 'Times New Roman');
        if row == 1
            title(ax, 'KF-AR time series comparison', 'Interpreter', 'none', 'FontName', 'Times New Roman');
            legend(ax, 'Location', 'best', 'Interpreter', 'latex', 'FontName', 'Times New Roman');
        end
        if row == numel(severities)
            xlabel(ax, 'Time [s]', 'Interpreter', 'latex', 'FontName', 'Times New Roman');
        end
    end

    % save
    out = fullfile('results', sprintf('%s_kf_ar_time_series_geometry', mdl));
    exportgraphics(fig,[out,'.pdf'],'ContentType','vector');
    savefig(fig,[out,'.fig']);
end

%% Auxiliary functions
function [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, ar_phase_idx] = ...
    get_overall_cfgs(cache_dir, scint_model, severity, is_refractive_effects_removed_received_signal, sigma2_W_3, sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance)
    % Define overall settings for the KF framework setup

    % General parameters
    doppler_profile = [0, 1000, 0.94];
    L1_C_over_N0_dBHz = 42;

    % Parameters for training the AR models for scintillation phase
    training_simulation_time = 300;
    is_refractive_effects_removed_training_data = true; % Exclude the refractive effects
    is_unwrapping_used = false; % This flag forces to use the wrapped phase for training the AR model

    % Parts for building the received signal
    cpssm_first_part = {L1_C_over_N0_dBHz, 'TPPSM', doppler_profile};
    cpssm_second_part = {'simulation_time', simulation_time, 'settling_time', settling_time, 'is_refractive_effects_removed', is_refractive_effects_removed_received_signal};

    % CPSSM timing scaling parameters for weak and strong scintillation
    rhof_veff_ratio_preset = [1.5, 0.27]; % See right plot of Table 3.2 of my dissertation.

    switch severity
        case 'weak'
            train_cfg_cpssm  = struct('scintillation_model', 'TPPSM', 'scenario', severity, ...
                'rhof_veff_ratio', rhof_veff_ratio_preset(1), ...
                'simulation_time', training_simulation_time, ...
                'is_refractive_effects_removed', is_refractive_effects_removed_training_data, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used, ...
                'is_enable_cmd_print', false);
            switch scint_model
                case 'cpssm'
                    ar_model_order = 14; % See right plot of Figure 4.7 of my dissertation.
                    rx_signal_model_inputs = [cpssm_first_part(:)',{seed},{'tppsm_scenario'}, {'weak'},cpssm_second_part(:)', 'rhof_veff_ratio', rhof_veff_ratio_preset(1)];
            end
        case 'strong'
            train_cfg_cpssm  = struct('scintillation_model', 'TPPSM', 'scenario', severity, ...
                'rhof_veff_ratio', rhof_veff_ratio_preset(2), ...
                'simulation_time', training_simulation_time, ...
                'is_refractive_effects_removed', is_refractive_effects_removed_training_data, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used, ...
                'is_enable_cmd_print', false);
            switch scint_model
                case 'cpssm' 
                    ar_model_order = 1; % See right plot of Figure 4.7 of my dissertation.
                    rx_signal_model_inputs = [cpssm_first_part(:)',{seed},{'tppsm_scenario'},{'strong'},cpssm_second_part(:)', 'rhof_veff_ratio', rhof_veff_ratio_preset(2)];
            end
    end

    expected_doppler_profile = [0,1000,0.94];

    gen_cfg_cpssm_base = struct( ...
      'kf_type', 'standard', ...
      'discrete_wiener_model_config', { {1, 3, sampling_interval, [1e-2, 0, sigma2_W_3], 1} }, ...
    'scintillation_training_data_config', train_cfg_cpssm, ...
      'C_over_N0_array_dBHz', L1_C_over_N0_dBHz, ...
      'initial_states_distributions_boundaries', { {[-pi, pi], [-25, 25], [-1, 1]} }, ...
      'expected_doppler_profile', expected_doppler_profile, ...
      'augmentation_model_initializer', struct('id', 'aryule', 'model_params', struct('model_order', ar_model_order)), ...
      'is_use_cached_settings', false, ...
      'is_generate_random_initial_estimates', true, ...
      'is_enable_cmd_print', false ...
    );

    gen_cfg_cpssm_geo = gen_cfg_cpssm_base;
    gen_cfg_cpssm_geo.scintillation_training_data_config = train_cfg_cpssm;
    gen_cfg_cpssm_geo.geometry_aided_measurement_config = struct('is_used', true, 'noise_variance', geometry_noise_variance);

    gen_cfg_cpssm_no_geo = gen_cfg_cpssm_base;
    gen_cfg_cpssm_no_geo.scintillation_training_data_config = train_cfg_cpssm;
    is_enable_cmd_print = false;

    [gen_kf_cfg_geo, init_estimates_cpssm_geo] = get_kalman_pll_config(gen_cfg_cpssm_geo, cache_dir, is_enable_cmd_print);
    [gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo] = get_kalman_pll_config(gen_cfg_cpssm_no_geo, cache_dir, is_enable_cmd_print);

    ar_phase_idx = length(expected_doppler_profile) + 1;
end

function [kf_ar_cfg, online_mdl_learning_cfg] = get_adaptive_cfgs()
    % Define approaches settings
    sampling_interval = 1e-2;

    % KF-AR without adaptation.
    kf_ar_cfg = struct(...
        'measurement_cov_adapt_algorithm', 'none', ...
        'states_cov_adapt_algorithm', 'none', ...
        'sampling_interval', sampling_interval, ...
        'hard_limited', struct('is_used', false, 'L1_C_over_N0_dBHz_threshold', NaN));
    
    % Online model learning setting
    online_mdl_learning_cfg = struct('is_online', false);
end
