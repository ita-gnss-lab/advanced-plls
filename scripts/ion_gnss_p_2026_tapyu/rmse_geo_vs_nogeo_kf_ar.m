% Description:
%   Script for running Monte Carlo runs for performance assessement
%   of the KF and KF-AR approaches for the CPSSM, with and without refractive effects
%   and with different values of the Wiener state noise variance (\sigma^2_{W,3}).
%   It is also plotted the performence of the geometry-aided KF-AR for
%   a fixed value of \sigma^2_{W,3}.
%
% Author: Rubem Vasconcelos Pacelli
% Email: rubem.engenharia@gmail.com

%% Initialization

clearvars; clc;

addpath(genpath(fullfile(pwd, '..', '..', 'libs')));

%inputs = get_signal_model_inputs('csm', 'strong', 1);
cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');

[kf_ar_cfg, ~, ~, ~, ~, online_mdl_learning_cfg] = get_adaptive_cfgs();

% Wiener state noise variance (\sigma^2_{W,3})
sigma2_W_3_sweep_amount = 10;
sigma2_W_3_sweep_no_geo = logspace(-14,2,sigma2_W_3_sweep_amount);
sigma2_W_3_sweep_geo = 1e-14 * ones(1, sigma2_W_3_sweep_amount); %logspace(-14,-10,sigma2_W_3_sweep_amount); % FIXME: using the same values for the geometry-aided until I have a better understanding of what σ^2_{W,3} models for the geometry-aided case.
% Amount of Monte Carlo runs
mc_runs = 10;
% Ionospheric Scintillation Severities
severities = ["weak", "strong"];

% General parameters
% Correlation sampling interval
sampling_interval = 1e-2; % 100 ms
settling_time = 50; % the start of the scintillation effects
simulation_time = 300;
geometry_noise_variance = 1e-12; % FIXME: you should adjust this value according to the expected accuracy of the satellite ephemerides

% Valid time vector (after the settling period)
valid_samples_idxs = ((settling_time/sampling_interval + 1) : simulation_time/sampling_interval).';

% Pre-allocate the results structure
results_matrix_template = zeros(sigma2_W_3_sweep_amount,mc_runs); % [sigma2_W_3_idx, seed]
struct_states = struct('phi_T', results_matrix_template, 'phi_W', results_matrix_template);
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
total_cppsm_wo_refr = numel(severities) * sigma2_W_3_sweep_amount * mc_runs;
iter = 0;
start_time = tic;
for severity = severities
    for sigma2_W_3_idx = 1:sigma2_W_3_sweep_amount
        for seed = 1:mc_runs
            [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
                get_overall_cfgs(cache_dir, 'cpssm', severity, true, sigma2_W_3_sweep_no_geo(sigma2_W_3_idx), sigma2_W_3_sweep_geo(sigma2_W_3_idx), sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance);

            % get received signal
            % NOTE: `diffractive_phase` is wrapped
            [rx_sig_cpssm_wo_refr, true_los_phase, ~, diffractive_phase] = get_received_signal(rx_signal_model_inputs{:});
            geometry_los_phase = get_noisy_geometry_phase(true_los_phase, sigma2_W_3_sweep_geo(sigma2_W_3_idx));
            
            % geometry-aided KF-AR
            [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
            % non-geometry-aided KF-AR
            [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 2. AKF-AR    : NWPR adaptive update with hard_limited = false.
            % TODO: [akf_ar_cpssm, ~]  = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', akf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 3. AHL-KF-AR : NWPR adaptive update with hard_limited = true.
            % TODO: [ahl_kf_ar_cpssm, ~]   = get_kalman_pll_estimates(rx_sig_cpssm_wo_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', ~, online_mdl_learning_cfg);

            % Define valid vectors, i.e., after the settling time
            valid_los_phase = true_los_phase(valid_samples_idxs,1); % This is being iterated unecessarily.
            valid_cpssm_phase = diffractive_phase(valid_samples_idxs,1);
            valid_total_phase = valid_los_phase + valid_cpssm_phase;

            %%% Extracting valid estimates, i.e., after the settling time
            % KF-AR non-geometry-aided
            kf_ar_no_geo_valid_hat_phi_W = kf_ar_cpssm_no_geo(valid_samples_idxs,1);
            kf_ar_no_geo_valid_hat_phi_AR = kf_ar_cpssm_no_geo(valid_samples_idxs,ar_phase_idx);
            kf_ar_no_geo_valid_hat_phi_T = kf_ar_no_geo_valid_hat_phi_W + kf_ar_no_geo_valid_hat_phi_AR;
            % KF-AR geometry-aided
            kf_ar_geo_valid_hat_phi_W = kf_ar_cpssm_geo(valid_samples_idxs,1);
            kf_ar_geo_valid_hat_phi_AR = kf_ar_cpssm_geo(valid_samples_idxs,ar_phase_idx);
            kf_ar_geo_valid_hat_phi_T = kf_ar_geo_valid_hat_phi_W + kf_ar_geo_valid_hat_phi_AR;
            % % AKF-AR
            % akf_ar_valid_hat_phi_W = akf_ar_cpssm(valid_samples_idxs,1);
            % akf_ar_valid_hat_phi_AR = akf_ar_cpssm(valid_samples_idxs,ar_phase_idx);
            % akf_ar_valid_hat_phi_T = akf_ar_valid_hat_phi_W + akf_ar_valid_hat_phi_AR;
            % % AHL-KF-AR
            % ahl_kf_ar_valid_hat_phi_W = ahl_kf_ar_cpssm(valid_samples_idxs,1);
            % ahl_kf_ar_valid_hat_phi_AR = ahl_kf_ar_cpssm(valid_samples_idxs,ar_phase_idx);
            % ahl_kf_ar_valid_hat_phi_T = ahl_kf_ar_valid_hat_phi_W + ahl_kf_ar_valid_hat_phi_AR;
            % % KF
            % kf_valid_hat_phi_T = kf_cpssm(valid_samples_idxs,1);
            % % AKF
            % akf_valid_hat_phi_T = akf_cpssm(valid_samples_idxs,1);

            %%% Saving the performance assessment
            % KF-AR non-geometry-aided
            results.cpssm_wo_refr.(severity).kf_ar_no_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_T - valid_total_phase));
            results.cpssm_wo_refr.(severity).kf_ar_no_geo.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_W - valid_los_phase));
            results.cpssm_wo_refr.(severity).kf_ar_no_geo.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_AR - valid_cpssm_phase));
            % KF-AR geometry-aided
            results.cpssm_wo_refr.(severity).kf_ar_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_T - valid_total_phase));
            results.cpssm_wo_refr.(severity).kf_ar_geo.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_W - valid_los_phase));
            results.cpssm_wo_refr.(severity).kf_ar_geo.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_AR - valid_cpssm_phase));
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
                    iter, total_cppsm_wo_refr, severity, sigma2_W_3_idx, sigma2_W_3_sweep_amount, seed, mc_runs, elapsed, remain);
        end
    end
end

%% CPSSM loop (diffractive + refractive phase)

fprintf('\nStarting CPSSM (wit refractive effect) simulations...\n');
total_cppsm_w_refr = numel(severities) * sigma2_W_3_sweep_amount * mc_runs;
iter = 0;
start_time = tic;
for severity = severities
    for sigma2_W_3_idx = 1:sigma2_W_3_sweep_amount
        for seed = 1:mc_runs
            [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
                get_overall_cfgs(cache_dir, 'cpssm', severity, false, sigma2_W_3_sweep_no_geo(sigma2_W_3_idx), sigma2_W_3_sweep_geo(sigma2_W_3_idx), sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance);

            [rx_sig_cpssm_w_refr, true_los_phase, psi_settled, diffractive_phase] = get_received_signal(rx_signal_model_inputs{:});
            geometry_los_phase = get_noisy_geometry_phase(true_los_phase, geometry_noise_variance);
            % For CPSSM, the training_scint_model is 'TPPSM' (AR augmented).
            % 1. KF-AR     : No adaptive update.
            [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
            [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 2. AKF-AR    : NWPR adaptive update with hard_limited = false.
            % TODO: [akf_ar_cpssm, ~]  = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', akf_ar_cfg, online_mdl_learning_cfg);
            % TODO: % 3. AHL-KF-AR : NWPR adaptive update with hard_limited = true.
            % TODO: [ahl_kf_ar_cpssm, ~]   = get_kalman_pll_estimates(rx_sig_cpssm_w_refr, gen_kf_cfg, init_estimates_cpssm, 'standard', 'TPPSM', ~, online_mdl_learning_cfg);
            
            % Define valid vectors, i.e., after the settling time
            valid_los_phase = true_los_phase(valid_samples_idxs,1); % This is being iterated unecessarily.
            valid_cpssm_diffractive_phase = diffractive_phase(valid_samples_idxs,1);
            valid_cpssm_total_phase = get_corrected_phase(psi_settled(valid_samples_idxs,1));
            valid_total_phase = valid_los_phase + valid_cpssm_total_phase;

            %%% Extracting valid estimates, i.e., after the settling time
            % KF-AR non-geometry-aided
            kf_ar_no_geo_valid_hat_phi_W = kf_ar_cpssm_no_geo(valid_samples_idxs,1);
            kf_ar_no_geo_valid_hat_phi_AR = kf_ar_cpssm_no_geo(valid_samples_idxs,ar_phase_idx);
            kf_ar_no_geo_valid_hat_phi_T = kf_ar_no_geo_valid_hat_phi_W + kf_ar_no_geo_valid_hat_phi_AR;
            % KF-AR geometry-aided
            kf_ar_geo_valid_hat_phi_W = kf_ar_cpssm_geo(valid_samples_idxs,1);
            kf_ar_geo_valid_hat_phi_AR = kf_ar_cpssm_geo(valid_samples_idxs,ar_phase_idx);
            kf_ar_geo_valid_hat_phi_T = kf_ar_geo_valid_hat_phi_W + kf_ar_geo_valid_hat_phi_AR;
            % TODO: % AKF-AR
            % akf_ar_valid_hat_phi_W = akf_ar_cpssm(valid_samples_idxs,1);
            % akf_ar_valid_hat_phi_AR = akf_ar_cpssm(valid_samples_idxs,ar_phase_idx);
            % akf_ar_valid_hat_phi_T = akf_ar_valid_hat_phi_W + akf_ar_valid_hat_phi_AR;
            % TODO: % AHL-KF-AR
            % ahl_kf_ar_valid_hat_phi_W = ahl_kf_ar_cpssm(valid_samples_idxs,1);
            % ahl_kf_ar_valid_hat_phi_AR = ahl_kf_ar_cpssm(valid_samples_idxs,ar_phase_idx);
            % ahl_kf_ar_valid_hat_phi_T = ahl_kf_ar_valid_hat_phi_W + ahl_kf_ar_valid_hat_phi_AR;
            % TODO: % KF
            % kf_valid_hat_phi_T = kf_cpssm(valid_samples_idxs,1);
            % TODO: % AKF
            % akf_valid_hat_phi_T = akf_cpssm(valid_samples_idxs,1);
            
            %%% Saving the performance assessment
            % KF-AR non-geometry-aided
            results.cpssm_w_refr.(severity).kf_ar_no_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_T - valid_total_phase));
            results.cpssm_w_refr.(severity).kf_ar_no_geo.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_W - valid_los_phase));
            results.cpssm_w_refr.(severity).kf_ar_no_geo.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_AR - valid_cpssm_phase));
            % KF-AR geometry-aided
            results.cpssm_w_refr.(severity).kf_ar_geo.phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_T - valid_total_phase));
            results.cpssm_w_refr.(severity).kf_ar_geo.phi_W(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_W - valid_los_phase));
            results.cpssm_w_refr.(severity).kf_ar_geo.phi_AR(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_AR - valid_cpssm_phase));
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
                    iter, total_cppsm_w_refr, severity, sigma2_W_3_idx, sigma2_W_3_sweep_amount, seed, mc_runs, elapsed, remain);
        end
    end
end

%% Script finalization
save(fullfile('results/results_L1_assessment.mat'), "results", "sigma2_W_3_sweep_no_geo", "sigma2_W_3_sweep_geo");


