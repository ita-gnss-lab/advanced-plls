% errors_comparison_all_approaches.m
%
% Overlay phase error time series for KF-AR with and without geometry aid, for both weak and strong scintillation
% - the state vector noise variance for geometry-aided is 1e-14, and for non-geometry-aided is 1e-6 (same as the value used for the plot in Rodrigo's dissertation);
% - the scintillation model is CPSM with refractive effects
% - the tracking error time series are computer after the settling time, and zoomed in the interval [50, 150] seconds, where the scintillation effects are about to begin
% - the diffractive and refractive phase components are plotted separately, to check whether the coupling problem has occurred for the non-geometry-aided approach
%
% Author: Rubem Vasconcelos Pacelli
% Email: rubem.engenharia@gmail.com

clearvars; clc;
addpath(genpath(fullfile(pwd,'..','..','libs')));

%% 1) Simulation settings
seed               = 6;
rng(seed);
severities         = {'weak','strong'};   % severity loop
sampling_interval  = 1e-2;
settling_time      = 50;
simulation_time    = 300;
results_dir        = fullfile(fileparts(mfilename('fullpath')),'results');
sigma2_W_3_geo     = 1e-14; % fixed AR-noise variance for the geometry-aided approache
sigma2_W_3_nogeo   = 1e-6;  % fixed AR-noise variance for the non-geometry-aided approaches % NOTE: this is the same value as used for the plot in Rodrigo's dissertation
scint_model = 'cpssm';
refr_flag = true;

%% 2) Zoom window
settling_time         = 50;  % seconds, the ending of the settling time, where the scitillation effects are about to begin
zoom_end           = 150; % seconds
idx_full           = round(settling_time/sampling_interval) : round(zoom_end/sampling_interval); % indices corresponding to the zoom window
time_zoom          = settling_time : sampling_interval : zoom_end; % temporal support for the zoom window

%% 3) Approach configs
approaches = {'KF-AR', 'KF-AR geo-aided'};
[kf_ar_cfg, ~, ~, ~, ~, ~] = get_adaptive_cfgs();

%% 4) Build received signals & true phases
data = struct('time_zoom', time_zoom);
simulation_settings = struct();
for s = severities
    sev = s{1};
    % get overall configs
    [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] =...
        get_overall_cfgs(cache_dir, 'cpssm', severity, true, sigma2_W_3_sweep_no_geo(sigma2_W_3_idx), sigma2_W_3_sweep_geo(sigma2_W_3_idx), sampling_interval, settling_time, simulation_time, seed);
    
    % get received signal
    [rx_sig, true_los_phase, ~, diffr_phase, refr_phase] = get_received_signal(rx_signal_model_inputs{:});
    geometry_los_phase = get_noisy_geometry_phase(true_los_phase, sigma2_W_3_geo);
        
    % non-geometry-aided KF-AR
    [kf_ar_cpssm_no_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg);
    % save results for non-geometry-aided approach
    simulation_settings.cpssm_w_refr.(sev).nogeo.rx_sig = rx_sig;
    simulation_settings.cpssm_w_refr.(sev).nogeo.gen_cfg = gen_cfg;
    simulation_settings.cpssm_w_refr.(sev).nogeo.init_cpssm = init_cpssm;
    simulation_settings.cpssm_w_refr.(sev).nogeo.init_none = init_none;
    simulation_settings.cpssm_w_refr.(sev).nogeo.ar_idx = ar_idx;

    % geometry-aided KF-AR
    [kf_ar_cpssm_geo, ~] = get_kalman_pll_estimates(rx_sig, gen_kf_cfg_geo, init_estimates_cpssm_geo, 'standard', 'TPPSM', kf_ar_cfg, online_mdl_learning_cfg, [], geometry_los_phase);
    % save results for geometry-aided approach
    simulation_settings.cpssm_w_refr.(sev).geo.rx_sig = rx_sig;
    simulation_settings.cpssm_w_refr.(sev).geo.gen_cfg = gen_cfg;
    simulation_settings.cpssm_w_refr.(sev).geo.init_cpssm = init_cpssm;
    simulation_settings.cpssm_w_refr.(sev).geo.init_none = init_none;
    simulation_settings.cpssm_w_refr.(sev).geo.ar_idx = ar_idx;
    
    % Store true phases
    data.cpssm_w_refr.(sev).phi_los_true = true_los_phase(idx_full);
    data.cpssm_w_refr.(sev).phi_d        = diffr_phase(idx_full);
    data.cpssm_w_refr.(sev).phi_r        = refr_phase(idx_full);
end

%% 5) Run PLL simulations for all approaches and store estimates

for fig_i = 1:numel(model_list)
    mdl = model_list{fig_i};

    for col = 1:2
        sev = severities{col};
        rx = simulation_settings.(mdl).(sev).rx_sig;
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

%% Script finalization

save(fullfile(results_dir,'errors_comparison_all_prev_approaches.mat'),'data');

