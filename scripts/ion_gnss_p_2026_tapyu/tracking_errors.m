% errors_comparison_all_approaches.m
%
% Overlay phase error time series for KF-AR with and without geometry aid, for both weak and strong scintillation
% - the state vector noise variance for geometry-aided is 1e-14, and for non-geometry-aided is 1e-6 (same as the
%   value used for the plot in Rodrigo's dissertation);
% - the scintillation model is CPSM with and without refractive effects
% - the tracking error time series are computer after the settling time, and zoomed in the interval [50, 150]
%   seconds, where the scintillation effects are about to begin
% - the diffractive and refractive phase components are plotted separately, to check whether the coupling problem
%   has occurred for the non-geometry-aided approach
%
% Author: Rubem Vasconcelos Pacelli
% Email: rubem.engenharia@gmail.com

clearvars; clc;
addpath(genpath(fullfile(pwd,'..','..','libs')));

%% 1) Simulation settings
cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');
seed               = 6;
rng(seed);
severities         = {'weak','strong'};   % severity loop
sampling_interval  = 1e-2;
results_dir        = fullfile(fileparts(mfilename('fullpath')),'results');
sigma2_W_3_nogeo   = 1e-6;  % fixed AR-noise variance for the non-geometry-aided approaches % NOTE: this is the same value as used for the plot in Rodrigo's dissertation
geometry_noise_variance = 1e-12; % FIXME: you should adjust this value according to the expected accuracy of the satellite ephemerides
sigma2_W_3_geo     = 1e-14; % fixed AR-noise variance for the geometry-aided approache % FIXME: using the same values for the geometry-aided until I have a better understanding of what σ^2_{W,3} models for the geometry-aided case.
scint_model = 'cpssm';
refr_flag = true;

%% 2) Zoom window
simulation_time             = 300;
settling_time               = 50;  % seconds, the ending of the settling time, where the scitillation effects are about to begin
zoom_end                    = 150; % seconds, the end of the zoom window, to make the plots more not too wide
valid_samples_idxs          = round(settling_time/sampling_interval) : round(zoom_end/sampling_interval); % indices corresponding to the zoom window
temporal_support            = settling_time : sampling_interval : zoom_end; % temporal support for the zoom window

%% 3) Approach configs
approaches = {'KF-AR', 'KF-AR geo-aided'};
[kf_ar_cfg, ~, ~, ~, ~, online_mdl_learning_cfg] = get_adaptive_cfgs();
model_list = {'cpssm_wo_refr','cpssm_w_refr'};

%% 4) Build received signals & true phases
data = struct('temporal_support', temporal_support);
simulation_settings = struct();
for m = model_list
    mdl = m{1};
    switch mdl
        case 'cpssm_wo_refr'
            is_remove_refractive = true;
        case 'cpssm_w_refr'
            is_remove_refractive = false;
    end
    for s = severities
        sev = s{1};
        % get overall configs
        [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
            get_overall_cfgs(cache_dir, 'cpssm', sev, is_remove_refractive, sigma2_W_3_nogeo, sigma2_W_3_geo, sampling_interval, settling_time, simulation_time, seed, geometry_noise_variance);
        
        % get received signal
        [rx_sig, true_los_phase, ~, diffr_phase, refr_phase] = get_received_signal(rx_signal_model_inputs{:});
        geometry_los_phase = get_noisy_geometry_phase(true_los_phase, sigma2_W_3_geo);
            
        % non-geometry-aided KF-AR
        [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
        % save results for non-geometry-aided approach
        data.cpssm_w_refr.(sev).nogeo.rx_sig = rx_sig;

        % geometry-aided KF-AR
        [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
        % save results for geometry-aided approach
        phi_w_hat = kf_ar_cpssm_geo(:,1);
        kf_estimates(:,ar_idx)
        
        % Store true phases
        data.cpssm_w_refr.(sev).phi_los_true = true_los_phase(valid_samples_idxs);
        data.cpssm_w_refr.(sev).phi_d        = diffr_phase(valid_samples_idxs);
        data.cpssm_w_refr.(sev).phi_r        = refr_phase(valid_samples_idxs);
    end
end

%% Script finalization

save(fullfile(results_dir,'tracking_errors.mat'),'data');

