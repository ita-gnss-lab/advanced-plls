% Description:
%   Script for running Monte Carlo runs for performance assessement
%   of the KF and KF-AR approaches for the CPSSM, with and withoud geometry aiding,
%   with and without refractive effects, and with different values of the AR noise variance
%   (\sigma^2_{W,3}). For the seed is defined by `seed_to_save_time_series` to save
%   the all time series of the estimates are saved for later plotting.
%
% Author: Rubem Vasconcelos Pacelli
% Email: rubem.engenharia@gmail.com

%% Initialization
clearvars; clc;

%%% Libs and paths
addpath(genpath(fullfile(pwd, '..', '..', 'libs')));
cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');
outfile = fullfile(fileparts(mfilename('fullpath')), 'results', 'results_L1_assessment.mat');

%%% Constantes
L1_frequency = 1575.42e6; % L1 frequency in Hz
c0 = 299792458; % Speed of light in m/s

%%% Geometric error statistics
geometry_types = ["broadcast", "precise"];
phi_LOS_noise_variances = dictionary();
p = 0.95; % confidence level for the geometry-aiding noise variance, i.e., the geometric error should be [-1, 1] [m] with 95% confidences % SEE: my article on ION GNSS+2026 for more details
alpha = 1 - p; % significance level
for geometry_type = geometry_types
    if strcmp(geometry_type, "broadcast")
        max_geo_error = 1; % [m] 95% maximum variation of the geometric error % SEE: "uncertainty range" column in Table 1 of my article on ION GNSS+2026 for more details
    elseif strcmp(geometry_type, "precise")
        max_geo_error = 0.05; % [m]
    end
    geometry_noise_std = max_geo_error / qfuncinv(alpha/2); % [m] standard deviation of the geometric error, which is obtained from the confidence level using the inverse of the Q-function
    geometry_noise_variance = geometry_noise_std^2; % [m^2] variance of
    phi_LOS_noise_variances(geometry_type) = (2 * pi * L1_frequency / c0)^2 * geometry_noise_variance; % [rad^2]
end

%%% Get KF parameters
[kf_ar_cfg, ~, ~, ~, ~, online_mdl_learning_cfg] = get_adaptive_cfgs();

%%% Ionospheric scintillation parameters
sigma2_W_3_sweep_amount = 10; % Wiener state noise variance (\sigma^2_{W,3})
sigma2_W_3_sweep = logspace(-18,9,sigma2_W_3_sweep_amount);
severities = ["weak", "strong"]; % Ionospheric Scintillation Severities

%%% Output data structure
sig = struct('phi_T', [], 'phi_W', [], 'phi_AR', []);
rmse = struct('delta_phi_T', [], 'delta_phi_LOS', [], 'delta_phi_s', []);
struct_states = struct('sig', sig, 'rmse', rmse);

ground_truth = struct('phi_los_true', [], 'phi_cpssm_total', [], 'phi_cpssm_refractive', [], 'phi_cpssm_diffractive', []);

approaches_struct = struct('kf_ar_geo', struct_states, ...
                           'kf_ar_no_geo', struct_states, ...
                           'kf', struct_states, ...
                           'saved_seed', struct('ground_truth', ground_truth, 'estimates', struct_states));

severities_struct = struct('weak', approaches_struct, 'strong', approaches_struct);

geometry_struct = struct('cpssm_wo_refr', severities_struct, 'cpssm_w_refr', severities_struct, 'temporal_support', []);
data = struct('broadcast', geometry_struct, 'precise', geometry_struct);

%%% Timing parameters
sampling_interval             = 10e-3; % 10 ms
settling_time                 = 50; % start of the scintillation effects
simulation_time               = 300; % how long the simulation runs
zoom_end                      = 150; % seconds, the end of the zoom window, to make the plots more not too wide
valid_samples_idxs            = round(settling_time/sampling_interval) : round(zoom_end/sampling_interval); % valid sample indices to the zoom window, which starts right after the settling period, i.e., when the scintillation begins
data.temporal_support         = settling_time : sampling_interval : zoom_end; % temporal support for the zoom window

%%% Simulation parameters
mc_seeds                 = 10; % Monte Carlo runs
seed_to_save_time_series = 1; % save the time series of the estimates for this seed
total_runs = numel(fieldnames(data)) * numel(severities) * sigma2_W_3_sweep_amount * mc_seeds;
iter = 0;

%% Main loop

fprintf('\nStarting CPSSM with and without refractive effects for KF-AR with and without geometry-aiding...\n');
for geometry_type = geometry_types
    phi_LOS_noise_variance = phi_LOS_noise_variances(geometry_type);
    for is_remove_refractive_effects = [false, true]
        if is_remove_refractive_effects
            refractivity = 'cpssm_wo_refr';
        else
            refractivity = 'cpssm_w_refr';
        end
        
        for severity = severities
            for seed = 1:mc_seeds
                for sigma2_W_3_idx = 1:sigma2_W_3_sweep_amount
                    [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
                        get_overall_cfgs(cache_dir, 'cpssm', severity, is_remove_refractive_effects, sigma2_W_3_sweep(sigma2_W_3_idx), sigma2_W_3_sweep(sigma2_W_3_idx), sampling_interval, settling_time, simulation_time, seed, phi_LOS_noise_variance);

                    %%% Get received signal
                    [rx_sig, true_los_phase, psi_settled, diffractive_phase] = get_received_signal(rx_signal_model_inputs{:}); % NOTE: `diffractive_phase` is wrapped
                    geometry_los_phase = get_noisy_geometry_phase(true_los_phase, phi_LOS_noise_variance);
                    
                    %%% Get Kalman estimates
                    % non-geometry-aided KF-AR
                    [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
                    % geometry-aided KF-AR
                    [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);

                    % Define valid vectors, i.e., after the settling time
                    valid_los_phase = true_los_phase(valid_samples_idxs,1); % This is being iterated unecessarily.
                    valid_cpssm_diffractive_phase = diffractive_phase(valid_samples_idxs,1);
                    if is_remove_refractive_effects
                        valid_cpssm_total_phase = valid_cpssm_diffractive_phase;
                    else
                        valid_cpssm_total_phase = get_corrected_phase(psi_settled(valid_samples_idxs,1));
                        valid_cpssm_refractive_phase = valid_cpssm_total_phase - valid_cpssm_diffractive_phase;
                    end
                    valid_total_phase = valid_los_phase + valid_cpssm_total_phase;

                    %%% Extracting valid estimates, i.e., after the settling time
                    % non-geometry-aided KF-AR
                    kf_ar_no_geo_valid_hat_phi_W = kf_ar_cpssm_no_geo(valid_samples_idxs,1);
                    kf_ar_no_geo_valid_hat_phi_AR = kf_ar_cpssm_no_geo(valid_samples_idxs,ar_phase_idx);
                    kf_ar_no_geo_valid_hat_phi_T = kf_ar_no_geo_valid_hat_phi_W + kf_ar_no_geo_valid_hat_phi_AR;
                    % geometry-aided KF-AR
                    kf_ar_geo_valid_hat_phi_W = kf_ar_cpssm_geo(valid_samples_idxs,1);
                    kf_ar_geo_valid_hat_phi_AR = kf_ar_cpssm_geo(valid_samples_idxs,ar_phase_idx);
                    kf_ar_geo_valid_hat_phi_T = kf_ar_geo_valid_hat_phi_W + kf_ar_geo_valid_hat_phi_AR;

                    %%% Saving the RMSE performance assessment
                    % non-geometry-aided KF-AR
                    data.(geometry_type).(refractivity).(severity).kf_ar_no_geo.rmse.delta_phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_T - valid_total_phase));
                    data.(geometry_type).(refractivity).(severity).kf_ar_no_geo.rmse.delta_phi_LOS(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_W - valid_los_phase));
                    data.(geometry_type).(refractivity).(severity).kf_ar_no_geo.rmse.delta_phi_s(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_no_geo_valid_hat_phi_AR - valid_cpssm_total_phase));
                    % geometry-aided KF-AR
                    data.(geometry_type).(refractivity).(severity).kf_ar_geo.rmse.delta_phi_T(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_T - valid_total_phase));
                    data.(geometry_type).(refractivity).(severity).kf_ar_geo.rmse.delta_phi_LOS(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_W - valid_los_phase));
                    data.(geometry_type).(refractivity).(severity).kf_ar_geo.rmse.delta_phi_s(sigma2_W_3_idx, seed) = rms(wrapToPi(kf_ar_geo_valid_hat_phi_AR - valid_cpssm_total_phase));

                    %%%% Saving signals (if the seed is the one defined in `seed_to_save_time_series`)
                    if seed == seed_to_save_time_series
                        %%% Save the estimates
                        % non-geometry-aided KF-AR
                        data.(geometry_type).(refractivity).(severity).kf_ar_no_geo.sig.phi_T(:, end+1) = kf_ar_no_geo_valid_hat_phi_T;
                        data.(geometry_type).(refractivity).(severity).kf_ar_no_geo.sig.phi_W(:, end+1) = kf_ar_no_geo_valid_hat_phi_W;
                        data.(geometry_type).(refractivity).(severity).kf_ar_no_geo.sig.phi_AR(:, end+1) = kf_ar_no_geo_valid_hat_phi_AR;
                        % geometry-aided KF-AR
                        data.(geometry_type).(refractivity).(severity).kf_ar_geo.sig.phi_T(:, end+1) = kf_ar_geo_valid_hat_phi_T;
                        data.(geometry_type).(refractivity).(severity).kf_ar_geo.sig.phi_W(:, end+1) = kf_ar_geo_valid_hat_phi_W;
                        data.(geometry_type).(refractivity).(severity).kf_ar_geo.sig.phi_AR(:, end+1) = kf_ar_geo_valid_hat_phi_AR;

                        %%% Save the true phases
                        data.(geometry_type).(refractivity).(severity).ground_truth.phi_los_true = valid_los_phase;
                        if is_remove_refractive_effects
                            data.(geometry_type).(refractivity).(severity).ground_truth.phi_cpssm_total = valid_cpssm_total_phase;
                        else
                            data.(geometry_type).(refractivity).(severity).ground_truth.phi_cpssm_refractive = valid_cpssm_refractive_phase;
                            data.(geometry_type).(refractivity).(severity).ground_truth.phi_cpssm_diffractive = valid_cpssm_diffractive_phase;
                        end
                    end

                    %%% Progress print
                    iter = iter + 1;
                    fprintf('CPSSM, is refractive: %d, for non-geometry aided [%d/%d] severity=%s, seed=%d/%d, sigma2_W_3_idx=%d/%d\n', ...
                            is_remove_refractive_effects, iter, total_runs, severity, seed, mc_seeds, sigma2_W_3_idx, sigma2_W_3_sweep_amount);
                end
            end
        end
    end
end

%% Save results
save(outfile, "data", "sigma2_W_3_sweep");
fprintf('\nData saved to %s\n', outfile);


