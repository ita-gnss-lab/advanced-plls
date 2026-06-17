function [rx_signal_model_inputs, gen_kf_cfg_geo, gen_kf_cfg_no_geo, init_estimates_cpssm_geo, init_estimates_cpssm_no_geo, init_estimates_none, ar_phase_idx] = ...
    get_overall_cfgs(cache_dir, scint_model, severity, is_refractive_effects_removed_received_signal, sigma2_W_3_no_geo, sigma2_W_3_geo, sampling_interval, settling_time, simulation_time, seed, phi_LOS_noise_variance)
    % Define overall settings for the KF framework setup

    % General parameters
    doppler_profile = [0, 1000, 0.94];
    L1_C_over_N0_dBHz = 42;

    % Parameters for training the AR models for scintillation phase
    training_simulation_time = 300;
    is_refractive_effects_removed_training_data = false; % whether to exclude the refractive effects % NOTE: Rodrigo was using obtaining the AR coefficients only with the diffractive component of the scintillation to analyze the impact of the unmodeled refractive effects on the performance of the KF-AR. Here, we are using the total scintillation phase (i.e., including the refractive effects) for training the AR model.
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
                    ar_model_order = 1; % SEE: My article ION GNSS+2026
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
                    ar_model_order = 1; % SEE: My article ION GNSS+2026
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
      'discrete_wiener_model_config', { {1, 3, sampling_interval, [0, 0, sigma2_W_3_geo], 1} }, ...
      'scintillation_training_data_config', train_cfg_cpssm, ...
      'C_over_N0_array_dBHz', L1_C_over_N0_dBHz, ...
      'initial_states_distributions_boundaries', { {[-pi, pi], [-25, 25], [-1, 1]} }, ...
      'expected_doppler_profile', expected_doppler_profile, ...
      'augmentation_model_initializer', struct('id', 'aryule', 'model_params', struct('model_order', ar_model_order)), ...
      'is_use_cached_settings', false, ...
      'is_generate_random_initial_estimates', true, ...
      'is_enable_cmd_print', false, ...
      'geometry_aided_measurement_config', struct('is_used', true, 'noise_variance', phi_LOS_noise_variance) ...
    );

    % Non-geometry-aided configuration (peform some updates to the geometry-aided configuration)
    gen_cfg_cpssm_no_geo = gen_cfg_cpssm_geo;
    gen_cfg_cpssm_no_geo.discrete_wiener_model_config = {1, 3, sampling_interval, [0, 0, sigma2_W_3_no_geo], 1}; % Update the Wiener state noise variance for the non-geometry-aided configuration
    gen_cfg_cpssm_no_geo.geometry_aided_measurement_config = struct('is_used', false);

    % No augmentation configuration (for training_scint_model = 'none')
    gen_cfg_none = gen_cfg_cpssm_geo;
    gen_cfg_none.scintillation_training_data_config = train_cfg_none;
    gen_cfg_none.augmentation_model_initializer.id = 'none';
    gen_cfg_none.augmentation_model_initializer.model_params = struct();
    is_enable_cmd_print = false;

    % Get the initial estimates for geometry-aided, non-geometry-aided, and no-augmentation (training_scint_model = 'none') configurations.
    [gen_kf_cfg_no_geo, init_estimates_cpssm_no_geo] = get_kalman_pll_config(gen_cfg_cpssm_no_geo, cache_dir, is_enable_cmd_print);
    [gen_kf_cfg_geo, init_estimates_cpssm_geo] = get_kalman_pll_config(gen_cfg_cpssm_geo, cache_dir, is_enable_cmd_print);
    [~, init_estimates_none] = get_kalman_pll_config(gen_cfg_none, cache_dir, is_enable_cmd_print);

    ar_phase_idx = length(expected_doppler_profile) + 1; % The AR phase is the last state in the state vector, after the Doppler states.
end