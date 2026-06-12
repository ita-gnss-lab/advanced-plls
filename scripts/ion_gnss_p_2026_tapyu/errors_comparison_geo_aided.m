% errors_comparison_all_approaches.m
%
% Overlay phase error time series for five PLL approaches:
% KF-AR, AKF-AR, AHL-KF-AR, KF, AKF
% Replicates exactly the layout and styling of error_series_plots_vs_sigma2_W,
% but with a fixed sigma2_W_3=1e-6 and overlay of all approaches.
% Now includes diffractive phase on last row (".-" style) and refractive in last fig legend.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

clearvars; clc;
addpath(genpath(fullfile(pwd,'..','..','libs')));

%% 1) Simulation settings
seed               = 6;
rng(seed);
severities         = {'weak','strong'};   % severity loop
sampling_interval  = 1e-2;
settling_time      = 50;
simulation_time    = 300;
results_dir          = fullfile(fileparts(mfilename('fullpath')),'results');

sigma2_W_3         = 1e-6;               % fixed AR-noise variance

%% 2) Zoom window
zoom_start         = 50;  % seconds, the ending of the settling time, where the scitillation effects are about to begin
zoom_end           = 150; % seconds
idx_full           = round(zoom_start/sampling_interval) : round(zoom_end/sampling_interval); % indices corresponding to the zoom window
time_zoom          = zoom_start : sampling_interval : zoom_end; % temporal support for the zoom window

%% 3) Approach configs
title_names = {'KF-AR', 'KF-AR geo-aided'};
[cfg_kf_ar, cfg_akf_ar, cfg_ahl_kf_ar, cfg_kf, cfg_akf, online_mdl_learning_cfg] = get_adaptive_cfgs();
all_cfgs     = {cfg_kf_ar, cfg_akf_ar, cfg_ahl_kf_ar, cfg_kf, cfg_akf};

%% 4) Build received signals & true phases
data = struct('time_zoom', time_zoom);
simulation_settings = struct();
model_list = {'cpssm_w_refr'};
for m = model_list
    mdl = m{1};
    switch mdl
        case 'cpssm_wo_refr'
            scint = 'cpssm'; refr_flag = true;
        case 'cpssm_w_refr'
            scint = 'cpssm'; refr_flag = false;
    end
    for s = severities
        sev = s{1};
        [rx_in, gen_cfg, init_cpssm, init_none, ar_idx] = ...
            get_overall_cfgs(results_dir, sev, refr_flag, sigma2_W_3, ...
                sampling_interval, settling_time, simulation_time, seed);
        [rx, true_los, psi, diffr_phase, refr_phase] = get_received_signal(rx_in{:});
        
        % Store settings
        simulation_settings.(mdl).(sev).rx = rx;
        simulation_settings.(mdl).(sev).gen_cfg = gen_cfg;
        simulation_settings.(mdl).(sev).init_cpssm = init_cpssm;
        simulation_settings.(mdl).(sev).init_none = init_none;
        simulation_settings.(mdl).(sev).ar_idx = ar_idx;
        % Store true phases
        data.(mdl).(sev).phi_los_true = true_los(idx_full);
        data.(mdl).(sev).phi_d        = diffr_phase(idx_full);
        % fill refr_phase consistently
        if refr_flag || strcmp(mdl,'cpssm_w_refr')
            data.(mdl).(sev).phi_r = refr_phase(idx_full);
        else
            data.(mdl).(sev).phi_r = zeros(size(idx_full));
        end
    end
end

%% 5) Run PLL simulations for all approaches and store estimates

for fig_i = 1:numel(model_list)
    mdl = model_list{fig_i};

    for col = 1:2
        sev = severities{col};
        rx = simulation_settings.(mdl).(sev).rx;
        gen_cfg = simulation_settings.(mdl).(sev).gen_cfg;
        init_cpssm = simulation_settings.(mdl).(sev).init_cpssm;
        init_none = simulation_settings.(mdl).(sev).init_none;
        ar_idx = simulation_settings.(mdl).(sev).ar_idx;

        data.(mdl).(sev).estimates = struct();

        for ia = 1:numel(title_names)
            cfg = all_cfgs{ia};
            if ia <= 3
                init_est = init_cpssm;
                training_scint = 'TPPSM';
            else
                init_est = init_none;
                training_scint = 'none';
            end

            [kf_estimates, ~] = get_kalman_pll_estimates(rx, gen_cfg, init_est, ...
                'standard', training_scint, cfg, online_mdl_learning_cfg);

            kf_estimates = kf_estimates(idx_full, :); % zoom window

            approach_name = matlab.lang.makeValidName(title_names{ia});
            % save kf estimates
            phi_w_hat = kf_estimates(:,1);
            data.(mdl).(sev).estimates.(approach_name).phi_w_hat = phi_w_hat;
            data.(mdl).(sev).estimates.(approach_name).delta_phi_los = data.(mdl).(sev).phi_los_true - phi_w_hat;
            phi_t_true = data.(mdl).(sev).phi_los_true + data.(mdl).(sev).phi_d + data.(mdl).(sev).phi_r;
            
            if size(kf_estimates, 2) >= ar_idx
                % If AR states are present, store them
                phi_ar_hat = kf_estimates(:,ar_idx);
                data.(mdl).(sev).estimates.(approach_name).phi_ar_hat = phi_ar_hat;
                phi_t_hat = phi_w_hat + phi_ar_hat;
            else
                % otherwise, set to empty and total = phi_w
                data.(mdl).(sev).estimates.(approach_name).phi_ar_hat = [];
                phi_t_hat = phi_w_hat;
            end
            data.(mdl).(sev).estimates.(approach_name).delta_phi_t = phi_t_true - phi_t_hat;
        end
    end
end

save(fullfile(results_dir,'errors_comparison_all_prev_approaches.mat'),'data');


%% Auxiliary functions
function [rx_signal_model_inputs, gen_kf_cfg, init_estimates_cpssm, init_estimates_none, ar_phase_idx] = ...
    get_overall_cfgs(results_dir, severity, is_refractive_effects_removed_received_signal, sigma2_W_3, sampling_interval, settling_time, simulation_time, seed)
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
            ar_model_order = 14; % See right plot of Figure 4.7 of my dissertation.
            rx_signal_model_inputs = [cpssm_first_part(:)',{seed},{'tppsm_scenario'}, {'weak'},cpssm_second_part(:)', 'rhof_veff_ratio', rhof_veff_ratio_preset(1)];
        case 'strong'
            train_cfg_cpssm  = struct('scintillation_model', 'TPPSM', 'scenario', severity, ...
                'rhof_veff_ratio', rhof_veff_ratio_preset(2), ...
                'simulation_time', training_simulation_time, ...
                'is_refractive_effects_removed', is_refractive_effects_removed_training_data, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used, ...
                'is_enable_cmd_print', false);
            ar_model_order = 1; % See right plot of Figure 4.7 of my dissertation.
            rx_signal_model_inputs = [cpssm_first_part(:)',{seed},{'tppsm_scenario'},{'strong'},cpssm_second_part(:)', 'rhof_veff_ratio', rhof_veff_ratio_preset(2)];
    end

    train_cfg_none  = struct('scintillation_model', 'none', 'sampling_interval', sampling_interval);

    expected_doppler_profile = [0,1000,0.94];

    gen_cfg_cpssm = struct( ...
      'kf_type', 'standard', ...
      'discrete_wiener_model_config', { {1, 3, sampling_interval, [0, 0, sigma2_W_3], 1} }, ...
      'scintillation_training_data_config', train_cfg_cpssm, ...
      'C_over_N0_array_dBHz', L1_C_over_N0_dBHz, ...
      'initial_states_distributions_boundaries', { {[-pi, pi], [-25, 25], [-1, 1]} }, ...
      'expected_doppler_profile', expected_doppler_profile, ...
      'augmentation_model_initializer', struct('id', 'aryule', 'model_params', struct('model_order', ar_model_order)), ...
      'is_use_cached_settings', false, ...
      'is_generate_random_initial_estimates', true, ...
      'is_enable_cmd_print', false ...
    );

    gen_cfg_none = gen_cfg_cpssm;
    gen_cfg_none.scintillation_training_data_config = train_cfg_none;
    gen_cfg_none.augmentation_model_initializer.id = 'none';
    gen_cfg_none.augmentation_model_initializer.model_params = struct();
    is_enable_cmd_print = false;

    [~, init_estimates_cpssm] = get_kalman_pll_config(gen_cfg_cpssm, results_dir, is_enable_cmd_print);
    [gen_kf_cfg, init_estimates_none] = get_kalman_pll_config(gen_cfg_none, results_dir, is_enable_cmd_print);

    ar_phase_idx = length(expected_doppler_profile) + 1;
end

function [kf_ar_cfg, akf_ar_cfg, ahl_kf_ar_cfg, kf_cfg, akf_cfg, online_mdl_learning_cfg] = get_adaptive_cfgs()
    % Define approaches settings
    sampling_interval = 1e-2;

    % Hard-Limiting constraint threshold.
    lambda = 35;  
    
    % NWPR parameters
    T_bit = 1/50;
    M_nwpr = T_bit / sampling_interval;
    N_nwpr = 20;
    
    % For AR (KFAR) estimates:
    kf_ar_cfg = struct(...
        'measurement_cov_adapt_algorithm', 'none', ...
        'states_cov_adapt_algorithm', 'none', ...
        'sampling_interval', sampling_interval, ...
        'hard_limited', struct('is_used', false));
    
    % For AR (KFAR) using the nwpr adaptive update (AKF):
    akf_ar_cfg = struct(...
        'measurement_cov_adapt_algorithm', 'nwpr', ...
        'measurement_cov_adapt_algorithm_params', struct('N_nwpr', N_nwpr, 'M_nwpr', M_nwpr), ...
        'states_cov_adapt_algorithm', 'none', ...
        'sampling_interval', sampling_interval, ...
        'hard_limited', struct('is_used', false));
    
    % For AR (KFAR) with hard limiting enabled (AHL-KF):
    ahl_kf_ar_cfg = struct(...
        'measurement_cov_adapt_algorithm', 'nwpr', ...
        'measurement_cov_adapt_algorithm_params', struct('N_nwpr', N_nwpr, 'M_nwpr', M_nwpr), ...
        'states_cov_adapt_algorithm', 'none', ...
        'hl_start_time', 50, ...
        'sampling_interval', sampling_interval, ...
        'hard_limited', struct('is_used', true, 'L1_C_over_N0_dBHz_threshold', lambda));
    
    % For standard KF estimates (training_scint_model = 'none'):
    kf_cfg = struct(...
        'measurement_cov_adapt_algorithm', 'none', ...
        'states_cov_adapt_algorithm', 'none', ...
        'sampling_interval', sampling_interval, ...
        'hard_limited', struct('is_used', false));
    
    akf_cfg = struct(...
        'measurement_cov_adapt_algorithm', 'nwpr', ...
        'measurement_cov_adapt_algorithm_params', struct('N_nwpr', N_nwpr, 'M_nwpr', M_nwpr), ...
        'states_cov_adapt_algorithm', 'none', ...
        'sampling_interval', sampling_interval, ...
        'hard_limited', struct('is_used', false));
    
    % Online model learning setting
    online_mdl_learning_cfg = struct('is_online', false);
end