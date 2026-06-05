% geometry_aided.m
%
% Description:
%   Script for running Monte Carlo runs for performance assessement
%   of the approaches KF, AKF, KF-AR, AKF-AR, and AHL-KF-AR, for
%   different values of the Wiener state noise variance (\sigma^2_{W,3})
%   with the aid of the geometry information
%
% Author: Rubem Vasconcelos Pacelli
% Email: rubem.engenharia@gmail.com

%% Initialization

clearvars; clc;

addpath(genpath(fullfile(pwd, '..', '..', 'libs')));

%inputs = get_signal_model_inputs('csm', 'strong', 1);
cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');

[kf_ar_cfg, akf_ar_cfg, ahl_kf_ar_cfg, kf_cfg, akf_cfg, online_mdl_learning_cfg] = get_adaptive_cfgs();

% Wiener state noise variance (\sigma^2_{W,3})
sigma2_W_3_sweep_amount = 5; % HACK: 20; for test
sigma2_W_3_sweep = logspace(-14,2,sigma2_W_3_sweep_amount);
% Amount of Monte Carlo runs
mc_runs = 1; % HACK: 100; for test
% Ionospheric Scintillation Severities
severities = ["weak", "strong"];

% General parameters
% Correlation sampling interval
sampling_interval = 1e-2; % 100 ms
settling_time = 50;
simulation_time = 300;
geometry_noise_variance = 1e-12; % HACK: 1e-14; for test.

% Valid time vector (after the settling period)
valid_samples_vector = ((settling_time/sampling_interval + 1) : simulation_time/sampling_interval).';

% Pre-allocate the results structure
results_matrix_template = zeros(length(sigma2_W_3_sweep),mc_runs); % [sigma2_W_3_idx, seed]
struct_states = struct('phi_T', results_matrix_template, 'phi_W', results_matrix_template); % ???: understand why we have phi_T and phi_W. I got this from `single_freq_assessment.m`
struct_aug_states = struct('phi_T', results_matrix_template, 'phi_W', results_matrix_template, 'phi_AR', results_matrix_template);
approaches_struct = struct('kf_ar_geo', struct_aug_states, ...
                           'kf_ar_no_geo', struct_aug_states, ...
                           'kf', struct_states);
                            % TODO: 'akf_ar', struct_aug_states, ...
                            % TODO: 'ahl_kf_ar', struct_aug_states, ...
                            % TODO: 'akf', struct('phi_T',results_matrix_template, 'phi_W', results_matrix_template))
severities_struct = struct('weak', approaches_struct, 'strong',approaches_struct);
results = struct('cpssm_wo_refr', severities_struct, 'cpssm_w_refr', severities_struct);

%% CPSSM loop (diffractive phase only)

fprintf('\nStarting CPSSM (diffractive phase only, without refractive effect) simulations...\n');
total_cppsm_wo_refr = numel(severities) * length(sigma2_W_3_sweep) * mc_runs;
iter = 0;
start_time = tic;
for severity = severities
    for sigma2_W_3_idx = 1:length(sigma2_W_3_sweep)
        for seed = 1:mc_runs
            [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
                get_overall_cfgs(cache_dir, 'cpssm', severity, true, sigma2_W_3_sweep(sigma2_W_3_idx), sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance);

            [rx_sig_cpssm_wo_refr, true_los_phase, ~, diffractive_phase] = get_received_signal(rx_signal_model_inputs{:});
            geometry_los_phase = get_noisy_geometry_phase(true_los_phase, geometry_noise_variance);
            % For CPSSM, the training_scint_model is 'TPPSM' (AR augmented).
            % 1. KF-AR     : No adaptive update.
            % geometry-aided KF-AR estimates
            [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
            % KF-AR
            [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 2. AKF-AR    : NWPR adaptive update with hard_limited = false.
            % TODO: [akf_ar_cpssm, ~]  = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', akf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 3. AHL-KF-AR : NWPR adaptive update with hard_limited = true.
            % TODO: [ahl_kf_ar_cpssm, ~]   = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', ahl_kf_ar_cfg, online_mdl_learning_cfg);

            % Define validation vectors (assuming valid_samples_vector is defined)
            valid_los_phase = true_los_phase(valid_samples_vector,1); % This is being iterated unecessarily.
            valid_cpssm_phase = diffractive_phase(valid_samples_vector,1);
            valid_total_phase = valid_los_phase + valid_cpssm_phase;

            %%% Extracting estimates
            % KF-AR
            kf_ar_geo_valid_hat_phi_W = kf_ar_cpssm_geo(valid_samples_vector,1);
            kf_ar_geo_valid_hat_phi_AR = kf_ar_cpssm_geo(valid_samples_vector,ar_phase_idx);
            kf_ar_geo_valid_hat_phi_T = kf_ar_geo_valid_hat_phi_W + kf_ar_geo_valid_hat_phi_AR;
            kf_ar_no_geo_valid_hat_phi_W = kf_ar_cpssm_no_geo(valid_samples_vector,1);
            kf_ar_no_geo_valid_hat_phi_AR = kf_ar_cpssm_no_geo(valid_samples_vector,ar_phase_idx);
            kf_ar_no_geo_valid_hat_phi_T = kf_ar_no_geo_valid_hat_phi_W + kf_ar_no_geo_valid_hat_phi_AR;
            % % AKF-AR
            % akf_ar_valid_hat_phi_W = akf_ar_cpssm(valid_samples_vector,1);
            % akf_ar_valid_hat_phi_AR = akf_ar_cpssm(valid_samples_vector,ar_phase_idx);
            % akf_ar_valid_hat_phi_T = akf_ar_valid_hat_phi_W + akf_ar_valid_hat_phi_AR;
            % % AHL-KF-AR
            % ahl_kf_ar_valid_hat_phi_W = ahl_kf_ar_cpssm(valid_samples_vector,1);
            % ahl_kf_ar_valid_hat_phi_AR = ahl_kf_ar_cpssm(valid_samples_vector,ar_phase_idx);
            % ahl_kf_ar_valid_hat_phi_T = ahl_kf_ar_valid_hat_phi_W + ahl_kf_ar_valid_hat_phi_AR;
            % % KF
            % kf_valid_hat_phi_T = kf_cpssm(valid_samples_vector,1);
            % % AKF
            % akf_valid_hat_phi_T = akf_cpssm(valid_samples_vector,1);

            %%% Saving the performance assessment
            % KF-AR
            results.cpssm_wo_refr.(severity).kf_ar_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_T - valid_total_phase));
            results.cpssm_wo_refr.(severity).kf_ar_no_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_T - valid_total_phase));
            % % AKF-AR
            % results.cpssm_wo_refr.(severity).akf_ar.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_ar_valid_hat_phi_W - valid_los_phase));
            % results.cpssm_wo_refr.(severity).akf_ar.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_ar_valid_hat_phi_AR - valid_cpssm_phase));
            % results.cpssm_wo_refr.(severity).akf_ar.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_ar_valid_hat_phi_T - valid_total_phase));
            % % AHL-KF-AR
            % results.cpssm_wo_refr.(severity).ahl_kf_ar.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(ahl_kf_ar_valid_hat_phi_W - valid_los_phase));
            % results.cpssm_wo_refr.(severity).ahl_kf_ar.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(ahl_kf_ar_valid_hat_phi_AR - valid_cpssm_phase));
            % results.cpssm_wo_refr.(severity).ahl_kf_ar.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(ahl_kf_ar_valid_hat_phi_T - valid_total_phase));
            % % KF
            % results.cpssm_wo_refr.(severity).kf.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_valid_hat_phi_T - valid_total_phase));
            % results.cpssm_wo_refr.(severity).kf.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_valid_hat_phi_T - valid_los_phase));
            % % AKF
            % results.cpssm_wo_refr.(severity).akf.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_valid_hat_phi_T - valid_total_phase));
            % results.cpssm_wo_refr.(severity).akf.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_valid_hat_phi_T - valid_los_phase));

            % progress print
            iter = iter + 1;
            elapsed = toc(start_time);
            remain  = elapsed * (total_cppsm_wo_refr/iter - 1);
            fprintf('CPSSM (w/o refractive effect) [%d/%d] severity=%s, sigma2_W_3_idx=%d/%d, seed=%d/%d, elapsed=%.1fs, remaining~%.1fs\n', ...
                    iter, total_cppsm_wo_refr, severity, sigma2_W_3_idx, length(sigma2_W_3_sweep), seed, mc_runs, elapsed, remain);
        end
    end
end

%% CPSSM loop (diffractive + refractive phase)

fprintf('\nStarting CPSSM (wit refractive effect) simulations...\n');
total_cppsm_w_refr = numel(severities) * length(sigma2_W_3_sweep) * mc_runs;
iter = 0;
start_time = tic;
for severity = severities
    for sigma2_W_3_idx = 1:length(sigma2_W_3_sweep)
        for seed = 1:mc_runs
            [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
                get_overall_cfgs(cache_dir, 'cpssm', severity, false, sigma2_W_3_sweep(sigma2_W_3_idx), sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance);

            [rx_sig_cpssm_w_refr, true_los_phase, psi_settled, diffractive_phase] = get_received_signal(rx_signal_model_inputs{:});
            geometry_los_phase = get_noisy_geometry_phase(true_los_phase, geometry_noise_variance);
            % For CPSSM, the training_scint_model is 'TPPSM' (AR augmented).
            % 1. KF-AR     : No adaptive update.
            [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
            [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 2. AKF-AR    : NWPR adaptive update with hard_limited = false.
            % TODO: [akf_ar_cpssm, ~]  = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', akf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 3. AHL-KF-AR : NWPR adaptive update with hard_limited = true.
            % TODO: [ahl_kf_ar_cpssm, ~]   = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', ahl_kf_ar_cfg, online_mdl_learning_cfg);
            
            % Define validation vectors (assuming valid_samples_vector is defined)
            valid_los_phase = true_los_phase(valid_samples_vector,1); % This is being iterated unecessarily.
            valid_cpssm_diffractive_phase = diffractive_phase(valid_samples_vector,1);
            valid_cpssm_total_phase = get_corrected_phase(psi_settled(valid_samples_vector,1));
            valid_total_phase = valid_los_phase + valid_cpssm_total_phase;

            %%% Extracting estimates
            % KF-AR
            kf_ar_geo_valid_hat_phi_W = kf_ar_cpssm_geo(valid_samples_vector,1);
            kf_ar_geo_valid_hat_phi_AR = kf_ar_cpssm_geo(valid_samples_vector,ar_phase_idx);
            kf_ar_geo_valid_hat_phi_T = kf_ar_geo_valid_hat_phi_W + kf_ar_geo_valid_hat_phi_AR;
            kf_ar_no_geo_valid_hat_phi_W = kf_ar_cpssm_no_geo(valid_samples_vector,1);
            kf_ar_no_geo_valid_hat_phi_AR = kf_ar_cpssm_no_geo(valid_samples_vector,ar_phase_idx);
            kf_ar_no_geo_valid_hat_phi_T = kf_ar_no_geo_valid_hat_phi_W + kf_ar_no_geo_valid_hat_phi_AR;
            % TODO: % AKF-AR
            % akf_ar_valid_hat_phi_W = akf_ar_cpssm(valid_samples_vector,1);
            % akf_ar_valid_hat_phi_AR = akf_ar_cpssm(valid_samples_vector,ar_phase_idx);
            % akf_ar_valid_hat_phi_T = akf_ar_valid_hat_phi_W + akf_ar_valid_hat_phi_AR;
            % TODO: % AHL-KF-AR
            % ahl_kf_ar_valid_hat_phi_W = ahl_kf_ar_cpssm(valid_samples_vector,1);
            % ahl_kf_ar_valid_hat_phi_AR = ahl_kf_ar_cpssm(valid_samples_vector,ar_phase_idx);
            % ahl_kf_ar_valid_hat_phi_T = ahl_kf_ar_valid_hat_phi_W + ahl_kf_ar_valid_hat_phi_AR;
            % TODO: % KF
            % kf_valid_hat_phi_T = kf_cpssm(valid_samples_vector,1);
            % TODO: % AKF
            % akf_valid_hat_phi_T = akf_cpssm(valid_samples_vector,1);
            
            %%% Saving the performance assessment
            % KF-AR
            results.cpssm_w_refr.(severity).kf_ar_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_T - valid_total_phase));
            results.cpssm_w_refr.(severity).kf_ar_no_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_T - valid_total_phase));
            % TODO: % AKF-AR
            % results.cpssm_w_refr.(severity).akf_ar.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_ar_valid_hat_phi_W - valid_los_phase));
            % results.cpssm_w_refr.(severity).akf_ar.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_ar_valid_hat_phi_AR - valid_cpssm_diffractive_phase));
            % results.cpssm_w_refr.(severity).akf_ar.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_ar_valid_hat_phi_T - valid_total_phase));
            % TODO: % AHL-KF-AR
            % results.cpssm_w_refr.(severity).ahl_kf_ar.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(ahl_kf_ar_valid_hat_phi_W - valid_los_phase));
            % results.cpssm_w_refr.(severity).ahl_kf_ar.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(ahl_kf_ar_valid_hat_phi_AR - valid_cpssm_diffractive_phase));
            % results.cpssm_w_refr.(severity).ahl_kf_ar.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(ahl_kf_ar_valid_hat_phi_T - valid_total_phase));
            % TODO: % KF
            % results.cpssm_w_refr.(severity).kf.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_valid_hat_phi_T - valid_los_phase));
            % results.cpssm_w_refr.(severity).kf.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_valid_hat_phi_T - valid_total_phase));
            % TODO: % AKF
            % results.cpssm_w_refr.(severity).akf.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_valid_hat_phi_T - valid_los_phase));
            % results.cpssm_w_refr.(severity).akf.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(akf_valid_hat_phi_T - valid_total_phase));

            % progress print
            iter = iter + 1;
            elapsed = toc(start_time);
            remain  = elapsed * (total_cppsm_w_refr/iter - 1);
            fprintf('CPSSM (w/ refractive effect) [%d/%d] severity=%s, sigma2_W_3_idx=%d/%d, seed=%d/%d, elapsed=%.1fs, remaining~%.1fs\n', ...
                    iter, total_cppsm_w_refr, severity, sigma2_W_3_idx, length(sigma2_W_3_sweep), seed, mc_runs, elapsed, remain);
        end
    end
end

save(fullfile('results/geometry_aided.mat'), "results", "sigma2_W_3_sweep");

%% Auxiliary functions
function [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] = ...
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
                otherwise
                    error('Unsupported scintillation model: %s', scint_model);
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
                otherwise
                    error('Unsupported scintillation model: %s', scint_model);
            end
    end

    train_cfg_none  = struct('scintillation_model', 'none', 'sampling_interval', sampling_interval);

    expected_doppler_profile = [0,1000,0.94];

    % Geometry-aided configuration
    gen_cfg_cpssm_geo = struct( ...
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
    gen_cfg_cpssm_geo.scintillation_training_data_config = train_cfg_cpssm;
    gen_cfg_cpssm_geo.geometry_aided_measurement_config = struct('is_used', true, 'noise_variance', geometry_noise_variance);

    % Non-geometry-aided configuration
    gen_cfg_cpssm_no_geo = gen_cfg_cpssm_geo;
    gen_cfg_cpssm_no_geo.scintillation_training_data_config = train_cfg_cpssm;
    gen_cfg_cpssm_no_geo.geometry_aided_measurement_config = struct('is_used', false);

    % No augmentation configuration (for training_scint_model = 'none')
    gen_cfg_none = gen_cfg_cpssm_geo;
    gen_cfg_none.scintillation_training_data_config = train_cfg_none;
    gen_cfg_none.augmentation_model_initializer.id = 'none';
    gen_cfg_none.augmentation_model_initializer.model_params = struct();
    is_enable_cmd_print = false;

    % Get the initial estimates for geometry-aided, non-geometry-aided, and no-augmentation (training_scint_model = 'none') configurations.
    [gen_kf_cfg_geo, init_estimates_cpssm_geo] = get_kalman_pll_config(gen_cfg_cpssm_geo, cache_dir, is_enable_cmd_print);
    [gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo] = get_kalman_pll_config(gen_cfg_cpssm_no_geo, cache_dir, is_enable_cmd_print);
    [~, init_estimates_none] = get_kalman_pll_config(gen_cfg_none, cache_dir, is_enable_cmd_print);

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

function geometry_los_phase = get_noisy_geometry_phase(true_los_phase, geometry_noise_variance)
    geometry_los_phase = true_los_phase + sqrt(geometry_noise_variance) * randn(size(true_los_phase));
end

