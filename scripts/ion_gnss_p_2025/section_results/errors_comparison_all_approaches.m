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
addpath(genpath(fullfile(pwd,'..','..','..','libs')));

%% 1) Simulation settings
seed               = 6;
rng(seed);
severities         = {'weak','strong'};   % severity loop
sampling_interval  = 1e-2;
settling_time      = 50;
simulation_time    = 300;
cache_dir          = fullfile(fileparts(mfilename('fullpath')),'cache');

sigma2_W_3         = 1e-6;               % fixed AR-noise variance

%% 2) Zoom window
zoom_start         = 50;  % seconds
zoom_end           = 150; % seconds
idx_full           = round(zoom_start/sampling_interval) : round(zoom_end/sampling_interval);
time_zoom          = (zoom_start : sampling_interval : zoom_end).' ;

%% 3) Approach configs
title_names = {'KF-AR','AKF-AR','AHL-KF-AR','KF','AKF'};
[cfg_kf_ar, cfg_akf_ar, cfg_ahl_kf_ar, cfg_kf, cfg_akf, online_mdl_learning_cfg] = get_adaptive_cfgs();
all_cfgs     = {cfg_kf_ar, cfg_akf_ar, cfg_ahl_kf_ar, cfg_kf, cfg_akf};

%% 4) Build received signals & true phases
data = struct();
model_list = {'csm','cpssm_wo_refr','cpssm_w_refr'};
for m = model_list
    mdl = m{1};
    switch mdl
        case 'csm'
            scint = 'csm'; refr_flag = true;
        case 'cpssm_wo_refr'
            scint = 'cpssm'; refr_flag = true;
        case 'cpssm_w_refr'
            scint = 'cpssm'; refr_flag = false;
    end
    for s = severities
        sev = s{1};
        [rx_in, gen_cfg, init_csm, init_cpssm, init_none, ar_idx] = ...
            get_overall_cfgs(cache_dir, scint, sev, refr_flag, sigma2_W_3, ...
                             sampling_interval, settling_time, simulation_time, seed);
        [rx, true_los, psi, diffr_phase, refr_phase] = get_received_signal(rx_in{:});
        data.(mdl).(sev).rx          = rx;
        data.(mdl).(sev).true_los    = true_los;
        data.(mdl).(sev).diffr_phase = diffr_phase;
        % fill refr_phase consistently
        if refr_flag || strcmp(mdl,'cpssm_w_refr')
            data.(mdl).(sev).refr_phase = refr_phase;
        else
            data.(mdl).(sev).refr_phase = zeros(size(diffr_phase));
        end
        data.(mdl).(sev).gen_cfg  = gen_cfg;
        data.(mdl).(sev).init_csm = init_csm;
        data.(mdl).(sev).init_none= init_none;
        data.(mdl).(sev).ar_idx   = ar_idx;
    end
end

%% 5) Plot figures
field_labels = { ...
    '$\hat{\phi}_{\mathrm{T},1} [k\mid k-1] - \phi_{\mathrm{T},1}[k]$', ...
    '$\hat{\phi}_{\mathrm{W},1} [k\mid k-1] - \phi_{\mathrm{LOS},1}[k]$', ...
    '$\hat{\phi}_{\mathrm{AR},1}[k\mid k-1]$', ...
};
font_size = 14;
colors    = winter(numel(title_names));
legend_pos = {[1,1],[1,1],[1,1]};

for fig_i = 1:numel(model_list)
    mdl = model_list{fig_i};
    fig = figure('Units','normalized','Position',[0.05 0.05 0.9 1]); % ,'Color','w'
    tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    for row = 1:3
        for col = 1:2
            ax = nexttile; hold(ax,'on');
            sev = severities{col};
            rx = data.(mdl).(sev).rx;
            true_los_phase = data.(mdl).(sev).true_los(idx_full);
            diffractive_phase = data.(mdl).(sev).diffr_phase(idx_full);
            refractive_phase = data.(mdl).(sev).refr_phase(idx_full);
            gen_cfg = data.(mdl).(sev).gen_cfg;
            init_csm  = data.(mdl).(sev).init_csm;
            init_none = data.(mdl).(sev).init_none;
            ar_idx    = data.(mdl).(sev).ar_idx;

            for ia = 1:numel(title_names)
                cfg = all_cfgs{ia};
                if ia<=3
                    init_est = init_csm;
                    if strcmp(mdl,'csm')
                        training_scint='CSM';
                    else 
                        training_scint='TPPSM';
                    end
                else
                    init_est = init_none; training_scint='none';
                end
                [kf_estimates,~] = get_kalman_pll_estimates(rx, gen_cfg, init_est, ...
                            'standard', training_scint, cfg, online_mdl_learning_cfg);
                phi_w = kf_estimates(:,1);
                switch row
                    case 1
                        if ia<=3
                            phi_t = phi_w + kf_estimates(:,ar_idx); 
                        else 
                            phi_t = phi_w;
                        end
                            if strcmp(mdl, 'csm') || strcmp(mdl, 'cpssm_wo_refr')
                                err = phi_t(idx_full) - (true_los_phase + diffractive_phase);
                            else
                                err = phi_t(idx_full) - (true_los_phase + diffractive_phase + refractive_phase);
                            end
                    case 2
                        err = phi_w(idx_full) - true_los_phase;
                    case 3
                        if ia<=3
                            err = kf_estimates(idx_full,ar_idx);
                        else 
                            err = NaN(size(idx_full));
                        end
                end
                plot(ax, time_zoom, err, 'LineWidth',1.5,'Color',colors(ia,:), 'DisplayName',title_names{ia});
            end
            if row == 1
                plot(ax, time_zoom, zeros(size(idx_full)),'LineWidth',2.5,'Color','k','DisplayName','Target Error');
            end
            if strcmp(mdl, 'csm')
                if row == 3
                    h_diff = plot(ax, time_zoom, diffractive_phase, '-.','LineWidth',1.5,'Color','k','DisplayName','Diffractive Phase $\phi_{\mathrm{D}, 1} [k]$');
                    legend(ax, h_diff, 'Interpreter','latex','Location','best', 'FontName','Times New Roman');
                end
            end
            if strcmp(mdl, 'cpssm_wo_refr')
                if row == 3
                    h_diff = plot(ax, time_zoom, diffractive_phase, '-.','LineWidth',1.5,'Color','k','DisplayName','Diffractive Phase $\phi_{\mathrm{D}, 1} [k]$');
                    legend(ax, h_diff, 'Interpreter','latex','Location','best', 'FontName','Times New Roman');
                end
            end
            if strcmp(mdl, 'cpssm_w_refr')
                % … after you’ve plotted everything else but before hold(ax,'off'):
                if fig_i==3 && row==2
                    % plot refractive and grab its handle
                    h_refr = plot(ax, time_zoom, refractive_phase, ':', ...
                                'LineWidth',1.5, 'Color','k', ...
                                'DisplayName','Refractive Phase ($\phi_{\mathrm{R},1}[k]$)');
                    % now make a legend showing only that handle
                    legend(ax, h_refr, 'Interpreter','latex','Location','best', 'FontName','Times New Roman');
                end
                if row == 3
                    h_diff = plot(ax, time_zoom, diffractive_phase, '-.','LineWidth',1.5,'Color','k','DisplayName','Diffractive Phase $\phi_{\mathrm{D}, 1} [k]$');
                    legend(ax, h_diff, 'Interpreter','latex','Location','best', 'FontName','Times New Roman');
                end
            end
            hold(ax,'off');

            if row==legend_pos{fig_i}(1) && col==legend_pos{fig_i}(2)
                legend(ax,'Location','best','Interpreter','latex','FontName','Times New Roman');
            end
            if strcmp(sev, 'weak')
                title_name_final = 'Weak';
            elseif strcmp(sev, 'strong')
                title_name_final = 'Strong';
            end
            if row==1, title(ax,title_name_final,'Interpreter','latex','FontName','Times New Roman'); end
            if col==1, ylabel(ax,field_labels{row},'Interpreter','latex','FontName','Times New Roman'); end
            if row==3, xlabel(ax,'Time [s]','Interpreter','latex','FontName','Times New Roman'); end
            grid(ax,'on'); grid(ax,'minor');
            set(ax,'FontName','Times New Roman','TickLabelInterpreter','latex','FontSize',font_size);
        end
    end
    out = fullfile('results',sprintf('%s_phase_errors_approaches_comparison',mdl));
    exportgraphics(fig,[out,'.pdf'],'ContentType','vector');
    savefig(fig,[out,'.fig']);
end


%% Auxiliary functions
function [rx_signal_model_inputs, gen_kf_cfg, init_estimates_csm, init_estimates_cpssm, init_estimates_none, ar_phase_idx] = ...
    get_overall_cfgs(cache_dir, scint_model, severity, is_refractive_effects_removed_received_signal, sigma2_W_3, sampling_interval, settling_time, simulation_time, seed)
    % Define overall settings for the KF framework setup

    % General parameters
    doppler_profile = [0, 1000, 0.94];
    L1_C_over_N0_dBHz = 42;

    % Parameters for training the AR models for scintillation phase
    training_simulation_time = 300;
    is_refractive_effects_removed_training_data = true; % Exclude the refractive effects
    is_unwrapping_used = false; % This flag forces to use the wrapped phase for training the AR model

    % Parts for building the received signal
    csm_first_part = {L1_C_over_N0_dBHz, 'CSM', doppler_profile};
    csm_second_part = {'simulation_time', simulation_time, 'settling_time', settling_time};
    cpssm_first_part = {L1_C_over_N0_dBHz, 'TPPSM', doppler_profile};
    cpssm_second_part = {'simulation_time', simulation_time, 'settling_time', settling_time, 'is_refractive_effects_removed', is_refractive_effects_removed_received_signal};

    % CSM parameters for weak and strong scintillation
    S4_preset = [0.2, 0.9];
    tau0_preset = [1, 0.2];

    % CPSSM timing scaling parameters for weak and strong scintillation
    rhof_veff_ratio_preset = [1.5, 0.27]; % See right plot of Table 3.2 of my dissertation.

    switch severity
        case 'weak'
            train_cfg_csm = struct('scintillation_model', 'CSM', 'S4', S4_preset(1), 'tau0', tau0_preset(1), ...
                'simulation_time', training_simulation_time, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used);
            train_cfg_cpssm  = struct('scintillation_model', 'TPPSM', 'scenario', severity, ...
                'rhof_veff_ratio', rhof_veff_ratio_preset(1), ...
                'simulation_time', training_simulation_time, ...
                'is_refractive_effects_removed', is_refractive_effects_removed_training_data, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used, ...
                'is_enable_cmd_print', false);
            switch scint_model
                case 'csm'
                    ar_model_order = 6; % See right plot of Figure 4.2 of my dissertation.
                    rx_signal_model_inputs = [csm_first_part(:)',{seed},{'S4'},{S4_preset(1)},{'tau0'},{tau0_preset(1)},csm_second_part(:)'];
                case 'cpssm'
                    ar_model_order = 14; % See right plot of Figure 4.7 of my dissertation.
                    rx_signal_model_inputs = [cpssm_first_part(:)',{seed},{'tppsm_scenario'}, {'weak'},cpssm_second_part(:)', 'rhof_veff_ratio', rhof_veff_ratio_preset(1)];
            end
        case 'strong'
            train_cfg_csm = struct('scintillation_model', 'CSM', 'S4', S4_preset(2), 'tau0', tau0_preset(2), ...
                'simulation_time', training_simulation_time, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used);
            train_cfg_cpssm  = struct('scintillation_model', 'TPPSM', 'scenario', severity, ...
                'rhof_veff_ratio', rhof_veff_ratio_preset(2), ...
                'simulation_time', training_simulation_time, ...
                'is_refractive_effects_removed', is_refractive_effects_removed_training_data, ...
                'sampling_interval', sampling_interval, ...
                'is_unwrapping_used', is_unwrapping_used, ...
                'is_enable_cmd_print', false);
            switch scint_model
                case 'csm'
                    rx_signal_model_inputs = [csm_first_part(:)',{seed},{'S4'},{S4_preset(2)},{'tau0'},{tau0_preset(2)},csm_second_part(:)'];
                    ar_model_order = 5; % See right plot of Figure 4.2 of my dissertation.
                case 'cpssm' 
                    ar_model_order = 1; % See right plot of Figure 4.7 of my dissertation.
                    rx_signal_model_inputs = [cpssm_first_part(:)',{seed},{'tppsm_scenario'},{'strong'},cpssm_second_part(:)', 'rhof_veff_ratio', rhof_veff_ratio_preset(2)];
            end
    end

    train_cfg_none  = struct('scintillation_model', 'none', 'sampling_interval', sampling_interval);

    expected_doppler_profile = [0,1000,0.94];

    gen_cfg_csm = struct( ...
      'kf_type', 'standard', ...
      'discrete_wiener_model_config', { {1, 3, sampling_interval, [0, 0, sigma2_W_3], 1} }, ...
      'scintillation_training_data_config', train_cfg_csm, ...
      'C_over_N0_array_dBHz', L1_C_over_N0_dBHz, ...
      'initial_states_distributions_boundaries', { {[-pi, pi], [-25, 25], [-1, 1]} }, ...
      'expected_doppler_profile', expected_doppler_profile, ...
      'augmentation_model_initializer', struct('id', 'aryule', 'model_params', struct('model_order', ar_model_order)), ...
      'is_use_cached_settings', false, ...
      'is_generate_random_initial_estimates', true, ...
      'is_enable_cmd_print', false ...
    );

    gen_cfg_cpssm = gen_cfg_csm;
    gen_cfg_cpssm.scintillation_training_data_config = train_cfg_cpssm;
    
    gen_cfg_none = gen_cfg_csm;
    gen_cfg_none.scintillation_training_data_config = train_cfg_none;
    gen_cfg_none.augmentation_model_initializer.id = 'none';
    gen_cfg_none.augmentation_model_initializer.model_params = struct();
    is_enable_cmd_print = false;

    [~, init_estimates_csm] = get_kalman_pll_config(gen_cfg_csm, cache_dir, is_enable_cmd_print);
    [~, init_estimates_cpssm] = get_kalman_pll_config(gen_cfg_cpssm, cache_dir, is_enable_cmd_print);
    [gen_kf_cfg, init_estimates_none] = get_kalman_pll_config(gen_cfg_none, cache_dir, is_enable_cmd_print);

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