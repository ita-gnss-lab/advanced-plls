function [state_estimates, ...
    error_covariance_estimates, ...
    L1_c_over_n0_linear_estimates] = ...
    ...
    get_kalman_pll_estimates( ...
    received_signal, ...
    kalman_pll_config, ...
    initial_estimates, ...
    kf_type, ...
    training_scint_model, ...
    adaptive_config, ...
    online_mdl_learning_cfg, ...
    psi_tppsm, ...
    geometry_measurement)
% get_kalman_pll_estimates
%
% Generates Kalman filter state and error covariance estimates based on the
% received signal and the provided Kalman PLL configuration.
%
% Syntax:
%   [state_estimates, error_covariance_estimates] = get_kalman_pll_estimates(received_signal, kalman_pll_config, initial_estimates, training_scint_model, adaptive_config, online_mdl_learning_cfg)
%
% Description:
%   This function applies the Kalman filter algorithm iteratively to the
%   received signal to produce state estimates and error covariance estimates.
%   It uses the configuration settings (F, Q, H, R, W) stored in the field of
%   kalman_pll_config corresponding to the given training_scint_model.
%
%   The training_scint_model parameter can be one of:
%     'CSM'      - Use Augmentation Model trained with the Cornell 
%                  Scintillation Model (CSM).
%     'TPPSM'    - Use Augmentation Model trained with the Two Component 
%                  Power Law Phase Screen Model (TPPSM).
%     'none'     - No augmentation model).
%
%   In addition, an adaptive configuration is applied based on the provided
%   adaptive_config struct. The adaptive configuration is organized into several
%   parts:
%
%   1. Measurement Covariance Adaptation:
%      - Field: measurement_cov_adapt_algorithm
%        Allowed values: {'none', 'simplified', 'nwpr'}.
%          * 'none': No adaptive update is applied (the measurement noise covariance R is used directly).
%          * 'simplified': A simplified adaptive update is applied using the estimated carrier-to-noise ratio.
%              In this case, the field measurement_cov_adapt_algorithm_params must be provided as a struct
%              containing at least:
%                  - L1_C_over_N0_dBHz : double value representing the estimated carrier-to-noise ratio.
%          * 'nwpr': A NWPR-based update is used. When selected, measurement_cov_adapt_algorithm_params must include
%              additional fields, e.g., 'N_nwpr', 'M_nwpr' (both doubles).
%
%   2. State Covariance Adaptation:
%      - Field: states_cov_adapt_algorithm
%        Allowed values: {'none', 'matching'}.
%          * 'none': No state covariance adaptation is applied.
%          * 'matching': An adaptive update using the covariance matching method is applied.
%              In this case, the field states_cov_adapt_algorithm_params must be provided with:
%                  - method     : string, either 'IAE' or 'RAE'.
%                  - N_nwpr: double value.
%                  - M_nwpr: double value
%
%   3. Global Sampling Interval:
%      - Field: sampling_interval (double)
%        This field is used consistently across both measurement and state adaptations.
%
%   4. Hard Limiting:
%      - Field: hard_limited (struct)
%        This struct must contain:
%                  - is_used: logical scalar indicating if hard limiting is enabled.
%                  - L1_C_over_N0_dBHz_threshold: double scalar threshold value.
%        Note: If hard limiting (hard_limited.is_used is true) is enabled, then a measurement
%        covariance adaptation algorithm must be specified (i.e., measurement_cov_adapt_algorithm must not be 'none').
%
% Inputs:
%   received_signal      - Numeric 2D array (NxM) representing the received signal.
%                          (Only the first column is used.)
%   kalman_pll_config    - Struct containing the Kalman filter configuration.
%                          Must have a field named after training_scint_model.
%   initial_estimates    - Struct with fields:
%                              x_hat_init: Numeric column vector (initial state estimate)
%                              P_hat_init: Numeric square matrix (initial error covariance)
%   kf_type              - Char with one of the following KF types: {'standard', 
%                          'extended', 'unscented', 'cubature'}
%   training_scint_model - A string with the name of the scintillation model used to
%                          train the AR model. Options: {'CSM', 'TPPSM', 'none'}.
%   adaptive_config      - A struct with adaptive configuration options for the Kalman filter.
%                          It must include the following fields:
%         .measurement_cov_adapt_algorithm
%             - Allowed values: {'none','simplified','nwpr'}.
%             - If not 'none', then the field measurement_cov_adapt_algorithm_params must be provided,
%               containing the required parameters (see above).
%         .states_cov_adapt_algorithm
%             - Allowed values: {'none','matching'}.
%             - If not 'none', then the field states_cov_adapt_algorithm_params must be provided,
%               containing the required parameters (see above).
%         .sampling_interval
%             - A double specifying the global sampling interval.
%         .hard_limited
%             - A struct with the fields:
%                  is_used (logical) and L1_C_over_N0_dBHz_threshold (double).
%
%   online_mdl_learning_cfg - Struct with settings for the online model learning:
%       'is_online': Boolean flag that determines if online augmentation model learning is used.
%       For AR augmentation models ('arfit', 'aryule'):
%         'learning_method': String with one of the following methods: {'sliding_window', 'block_window', 'kalman'}.
%         'window_size' (optional): Window size for the chosen method.
%         'kalman_cfg' (optional): Kalman configuration for the online AR model parameter estimation.
%       For RBF augmentation model ('rbf'):
%         'rbf_cfg': RBF network configuration for online model learning.
%
% Outputs:
%   state_estimates            - Numeric matrix of state estimates. Each row corresponds
%                                to a time step; the number of columns equals the length of x_hat_init.
%   error_covariance_estimates - 3D numeric array containing error covariance estimates.
%                                Its first dimension is time, and the second and third dimensions
%                                match the size of P_hat_init.
%
% References: 
% [1] Vilà-Valls J, Closas P, Curran JT. 2017. Multi-frequency GNSS robust carrier tracking 
%     for ionospheric scintillation mitigation. J. Space Weather Space Clim. 7: A26
% [2] Florindo, Rodrigo de Lima, Antreich, Felix, "Multi-Frequency Kalman Filter Carrier 
%     Phase Tracking for Ionospheric Scintillation Mitigation and Monitoring," Proceedings
%     of the 37th International Technical Meeting of the Satellite Division of The Institute
%     of Navigation (ION GNSS+ 2024), Baltimore, Maryland, September 2024, pp. 3611-3625.
%     https://doi.org/10.33012/2024.19899
% [3] R. A. M. Lopes, F. Antreich, F. Fohlmeister, M. Kriegel and H. K. Kuga, "Ionospheric 
%     Scintillation Mitigation With Kalman PLLs Employing Radial Basis Function Networks," 
%     in IEEE Transactions on Aerospace and Electronic Systems, vol. 59, no. 5, pp. 6878-6893,
%     Oct. 2023, doi: 10.1109/TAES.2023.3281431
% [4] Locubirche-Serra, Sergi. Robust Carrier Tracking Techniques for GNSS Receivers affected by
%     Ionospheric Scintillation. Doctoral Thesis. https://ddd.uab.cat/record/235075?ln=en
% [5] Locubiche-Serra, S., Seco-Granados, G. & López-Salcedo, J.A. Performance assessment of a 
%     low-complexity autoregressive Kalman filter for GNSS carrier tracking using real 
%     scintillation time series. GPS Solut 26, 17 (2022). https://doi.org/10.1007/s10291-021-01193-0
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    if nargin < 8
        psi_tppsm = [];
    end
    if nargin < 9
        geometry_measurement = [];
    end
    
    [F, Q, H, R, W, Hj_handle, state_estimates, error_covariance_estimates, L1_c_over_n0_linear_estimates, extra_vars, N, ut_params] = ...
    initializer_function(received_signal, kalman_pll_config, initial_estimates, kf_type, training_scint_model, adaptive_config, online_mdl_learning_cfg, geometry_measurement);

    %% Main filtering loop.
    for step = 1:N
        if step > 1
            % Determine the measurement noise covariance adapt_R.
            [adapt_R, extra_vars, L1_c_over_n0_linear_estimates] = compute_adaptive_R(...
                step, received_signal, adaptive_config, extra_vars, H, R, L1_c_over_n0_linear_estimates);

            % Apply hard-limited constraint if enabled.
            if adaptive_config.hard_limited.is_used
                if step * adaptive_config.sampling_interval > adaptive_config.hl_start_time
                    if (10 * log10(extra_vars.latest_L1_c_over_n0_linear) < extra_vars.threshold)
                        % TODO: It is hard coded here. Doesn't fit the extended
                        % KF SSM neither the multi-frequency case.
                        adapt_R = 999;
                    end
                end
            end

            % Adapt the augmentation models in an online manner if enabled.
            if online_mdl_learning_cfg.is_online
                [F, Q, W] = update_filter_matrices(F, Q, W, step, state_estimates, online_mdl_learning_cfg, kalman_pll_config.(training_scint_model).(kf_type).augmentation_model_initializer);
            end

            % Variant-specific update.
            switch lower(kf_type)
                case 'standard'
                    [K, innovation, x_hat_update, P_hat_update] = standard_kf_update(...
                        step, received_signal, extra_vars, H, adapt_R, R);
                case 'extended'
                    [K, innovation, x_hat_update, P_hat_update] = extended_kf_update(...
                        step, received_signal, Hj_handle, extra_vars, adapt_R, R, psi_tppsm);
                case 'unscented'
                    [K, innovation, x_hat_update, P_hat_update] = unscented_kf_update(...
                        step, received_signal, extra_vars, adapt_R, ut_params, psi_tppsm);
                otherwise
                    error('get_kalman_pll_estimates:unsupported_kf', 'KF type %s is not yet supported.', kf_type);
            end

            % Adapt the state covariance (if using the matching algorithm).
            if strcmpi(adaptive_config.states_cov_adapt_algorithm, 'matching')
                extra_vars = update_state_covariance(step, received_signal, adaptive_config, extra_vars, Q, H, innovation, K);
                % extra_vars.P_hat (or Q) may be updated inside the above call.
                Q = extra_vars.updated_Q;
            end

        else
            x_hat_update = extra_vars.x_hat_project_ahead;
            P_hat_update = extra_vars.P_hat_project_ahead;
        end
        
        % Projection step.
        extra_vars.x_hat_project_ahead = F * x_hat_update + W;
        extra_vars.P_hat_project_ahead = F * P_hat_update * F.' + Q;
        
        % Save estimates.
        state_estimates(step, :) = extra_vars.x_hat_project_ahead.';
        error_covariance_estimates(step, :, :) = extra_vars.P_hat_project_ahead;
    end
end

function [F, Q, H, R, W, Hj_handle, state_estimates, error_covariance_estimates, L1_c_over_n0_linear_estimates, extra_vars, N, ut_params] = initializer_function(...
    received_signal, kalman_pll_config, initial_estimates, kf_type, training_scint_model, adaptive_config, online_mdl_learning_cfg, geometry_measurement)
% initializer_function
%
% Initializes the Kalman PLL estimation by validating inputs, retrieving the
% necessary filter matrices from the configuration, preallocating output arrays,
% and setting up additional variables for the filtering loop.
%
% Syntax:
%   [F, Q, H, R, W, Hj_handle, state_estimates, error_covariance_estimates, ...
%    L1_c_over_n0_linear_estimates, extra_vars, N] = ...
%         initializer_function(received_signal, kalman_pll_config, initial_estimates, ...
%                              kf_type, training_scint_model, adaptive_config, online_mdl_learning_cfg)
%
% Description:
%   This function performs validation of the provided inputs and then extracts the
%   filter matrices (F, Q, H, R, W) and the measurement Jacobian handle (Hj_handle)
%   from the kalman_pll_config structure based on the given training_scint_model and kf_type.
%   It preallocates the state estimates and error covariance arrays according to the
%   number of time steps (rows) in the received_signal and initializes an extra_vars
%   structure that contains:
%     - The projected state and error covariance (x_hat_project_ahead and P_hat_project_ahead)
%     - Thresholds and baseline parameters for measurement adaptation (for both 'simplified'
%       and 'nwpr' algorithms)
%     - Parameters for NWPR-based adaptation such as z_vec, NP_vec, and NP_array if applicable.
%
% Inputs:
%   received_signal         - Numeric 2D array (NxM) representing the received signal.
%                             (Only the first column is used.)
%   kalman_pll_config       - Struct containing the Kalman filter configuration. Must have a field 
%                             corresponding to the training_scint_model that includes matrices F, Q, H, R, and W.
%   initial_estimates       - Struct with fields:
%                                x_hat_init: Numeric column vector (initial state estimate)
%                                P_hat_init: Numeric square matrix (initial error covariance)
%   kf_type                 - Character or string specifying the Kalman filter type. Allowed values:
%                             {'standard', 'extended', 'unscented', 'cubature'}.
%   training_scint_model    - String with the name of the scintillation model used to train the AR model.
%                             Options: {'CSM', 'TPPSM', 'none'}.
%   adaptive_config         - Struct with adaptive configuration options for the Kalman filter.
%                             This includes settings for measurement and state covariance adaptation.
%   online_mdl_learning_cfg - Struct with settings for online model learning for the augmentation model.
%
% Outputs:
%   F                           - State transition matrix.
%   Q                           - Process noise covariance matrix.
%   H                           - Measurement transition matrix.
%   R                           - Measurement noise covariance matrix.
%   W                           - Bias vector used by the ARFIT augmentation model.
%   Hj_handle                   - Function handle for the measurement Jacobian (used in the extended KF).
%   state_estimates             - Preallocated numeric matrix to store state estimates.
%   error_covariance_estimates  - Preallocated 3D numeric array to store error covariance estimates.
%   L1_c_over_n0_linear_estimates - Preallocated numeric vector for storing carrier-to-noise ratio estimates.
%   extra_vars                  - Structure containing additional variables needed in the filtering loop, including:
%                                 x_hat_project_ahead, P_hat_project_ahead, thresholds, baseline values, and NWPR parameters.
%   N                           - Number of time steps, i.e., the number of rows in received_signal.
%
% References:
%   [1] Brown, Robert Grover, and Patrick Y C Hwang. “Introduction to 
%       Random Signals and Applied Kalman Filtering: With MATLAB® Exercises, 
%       Fourth Edition”
%   [2] E. A. Wan and R. Van Der Merwe, "The unscented Kalman filter for 
%       nonlinear estimation," Proceedings of the IEEE 2000 Adaptive 
%       Systems for Signal Processing, Communications, and Control 
%       Symposium (Cat. No.00EX373), Lake Louise, AB, Canada, 2000, 
%       pp. 153-158, doi: 10.1109/ASSPCC.2000.882463.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    % Validate inputs
    validate_received_signal(received_signal);
    validate_initial_estimates(initial_estimates);
    validate_kf_type(kf_type);
    validate_training_scint_model(training_scint_model, kalman_pll_config);
    validate_adaptive_config(adaptive_config);
    validate_online_learning_cfg(online_mdl_learning_cfg, kalman_pll_config, training_scint_model, kf_type)
    if nargin < 8
        geometry_measurement = [];
    end
    
    % Retrieve filter matrices from configuration.
    config_struct = kalman_pll_config.(training_scint_model).(kf_type);
    required_config_fields = {'F','Q','H','R','W'};
    for i = 1:length(required_config_fields)
        if ~isfield(config_struct, required_config_fields{i})
            error('initializeFilter:missing_config_field', ...
                'kalman_pll_config.%s is missing the field %s.', training_scint_model, required_config_fields{i});
        end
    end
    F = config_struct.F;
    Q = config_struct.Q;
    H = config_struct.H;
    R = config_struct.R;
    W = config_struct.W;
    Hj_handle = config_struct.Hj_handle;
    
    % Preallocate arrays
    N = size(received_signal, 1);
    states_amount = numel(initial_estimates.x_hat_init);
    state_estimates = zeros(N, states_amount);
    error_covariance_estimates = zeros(N, size(initial_estimates.P_hat_init, 1), size(initial_estimates.P_hat_init,2));
    L1_c_over_n0_linear_estimates = NaN(N, 1);
    
    % Set initial estimates for the filtering loop.
    extra_vars.x_hat_project_ahead = initial_estimates.x_hat_init;
    extra_vars.P_hat_project_ahead = initial_estimates.P_hat_init;
    extra_vars.is_geometry_aided = ~isempty(geometry_measurement);
    extra_vars.geometry_measurement = geometry_measurement(:);
    if extra_vars.is_geometry_aided
        if numel(extra_vars.geometry_measurement) ~= N
            error('get_kalman_pll_estimates:invalid_geometry_measurement_length', ...
                'Geometry measurement must have the same number of samples as received_signal.');
        end
        extra_vars.geometry_noise_variance = [];
        if isfield(kalman_pll_config.(training_scint_model).(kf_type), 'geometry_aided_measurement_config') && ...
                isfield(kalman_pll_config.(training_scint_model).(kf_type).geometry_aided_measurement_config, 'noise_variance')
            extra_vars.geometry_noise_variance = kalman_pll_config.(training_scint_model).(kf_type).geometry_aided_measurement_config.noise_variance;
        end
    end

    augmentation_model_id = lower(char(config_struct.augmentation_model_initializer.id));
    extra_vars.augmentation_model_id = augmentation_model_id;
    extra_vars.measurement_model = 'external_amplitude_phase';
    if strcmpi(kf_type, 'unscented') && strcmp(augmentation_model_id, 'arfit-complex-field')
        if ~strcmpi(adaptive_config.measurement_cov_adapt_algorithm, 'none') || ...
                ~strcmpi(adaptive_config.states_cov_adapt_algorithm, 'none')
            error('get_kalman_pll_estimates:unsupported_adaptive_complex_field', ...
                'The arfit-complex-field UKF path currently supports only nonadaptive filtering.');
        end
        model_order = config_struct.augmentation_model_initializer.model_params.model_order;
        n_los = states_amount - 2 * model_order;
        if n_los < 1
            error('get_kalman_pll_estimates:invalid_complex_field_state_layout', ...
                'The arfit-complex-field state vector does not contain a valid LOS block.');
        end
        extra_vars.measurement_model = 'complex_field';
        extra_vars.idx.phase = 1;
        extra_vars.idx.field_real = n_los + 1;
        extra_vars.idx.field_imag = n_los + 2;
    end
    
    % Initialize any other variables that are shared in the loop,
    % such as thresholds or baseline measurements from the adaptive_config.
    if adaptive_config.hard_limited.is_used
        extra_vars.threshold = adaptive_config.hard_limited.L1_C_over_N0_dBHz_threshold;
    end
    
    % Setup additional baseline parameters for the measurement adaptation.
    if strcmpi(adaptive_config.measurement_cov_adapt_algorithm, 'simplified')
        extra_vars.baseline_L1_C_over_N0_dBHz = adaptive_config.measurement_cov_adapt_algorithm_params.L1_C_over_N0_dBHz;
        extra_vars.baseline_L1_c_over_n0_linear = 10^(extra_vars.baseline_L1_C_over_N0_dBHz / 10);
        extra_vars.sampling_interval = adaptive_config.sampling_interval;
    end

    if strcmpi(adaptive_config.measurement_cov_adapt_algorithm, 'nwpr')
        extra_vars.N_nwpr = adaptive_config.measurement_cov_adapt_algorithm_params.N_nwpr;
        extra_vars.M_nwpr = adaptive_config.measurement_cov_adapt_algorithm_params.M_nwpr;
        extra_vars.sampling_interval = adaptive_config.sampling_interval;
        extra_vars.z_vec = received_signal(1:extra_vars.M_nwpr);
        % Compute the initial NP and setup NP_vec and NP_array.
        NBP = (sum(real(extra_vars.z_vec)))^2 + (sum(imag(extra_vars.z_vec)))^2;
        WBP = sum(abs(extra_vars.z_vec).^2);
        current_NP = NBP / WBP;
        vec_size = extra_vars.N_nwpr / extra_vars.M_nwpr;
        extra_vars.NP_vec = repmat(current_NP, vec_size, 1);
        extra_vars.NP_array = zeros(N, 1);
    end

    % Initialize the states error memory used in the covariance
    % matching adaptive state covariance algorithm.
    if strcmpi(adaptive_config.states_cov_adapt_algorithm, 'matching')
        extra_vars.states_error_mem = zeros(adaptive_config.states_cov_adapt_algorithm_params.window_size,1);
    end

    ut_params = struct();
    if strcmpi(kf_type, 'unscented')
        % NOTE: Given that [1, Chapter 7 - The unscented Kalman filter]
        % do not explain with details which unscented transform parameters 
        % should be generally adopted, We refer to the parameter 
        % specification provided in [2] instead.
        alpha = 1e-3;
        beta = 2; % Optimal for Gaussian distributions
        kappa = 3 - states_amount;
        lambda = alpha^2 * (states_amount + kappa) - states_amount;
        ut_params.lambda = lambda;
        ut_params.omega_mean_0 = lambda / (lambda + states_amount);
        ut_params.omega_covariance_0 = ut_params.omega_mean_0 + 1 - alpha^2 + beta;
        ut_params.omega_mean_covariance_general = 1 / (2*(lambda + states_amount));
        ut_params.sigma_points_amount = 2*states_amount + 1;
        ut_params.states_amount = states_amount;
    end
end

function [adapt_R, extra_vars, L1_c_over_n0_linear_estimates] = compute_adaptive_R(...
    step, received_signal, adaptive_config, extra_vars, H, R, L1_c_over_n0_linear_estimates)
% compute_adaptive_R
%
% Computes the adaptive measurement noise covariance (adapt_R) based on the current
% measurement adaptation algorithm specified in adaptive_config. This function supports
% different adaptation strategies ('none', 'simplified', and 'nwpr') and updates the
% provided extra_vars structure with any additional parameters required for adaptation,
% such as the latest carrier-to-noise ratio estimates.
%
% Syntax:
%   [adapt_R, extra_vars, L1_c_over_n0_linear_estimates] = compute_adaptive_R(...
%         step, received_signal, adaptive_config, extra_vars, H, R, L1_c_over_n0_linear_estimates)
%
% Description:
%   Depending on the value of adaptive_config.measurement_cov_adapt_algorithm, this
%   function computes:
%     - For 'none': Returns the default fixed measurement noise covariance R.
%     - For 'simplified': Estimates the carrier-to-noise ratio from the signal intensity
%       and computes the phase noise variance as adapt_R. It also stores the current estimate
%       in L1_c_over_n0_linear_estimates and extra_vars.
%     - For 'nwpr': Uses a NWPR-based update where it computes the normalization factor (NP)
%       from a sliding window of measurements stored in extra_vars.z_vec. The function updates
%       extra_vars (including NP_vec and NP_array) and L1_c_over_n0_linear_estimates accordingly.
%
% Inputs:
%   step                        - Scalar indicating the current time step (iteration index).
%   received_signal             - Numeric 2D array (NxM) representing the received signal.
%   adaptive_config             - Struct containing adaptive configuration options. This includes
%                                 the measurement adaptation algorithm and any required parameters.
%   extra_vars                  - Structure that holds additional variables for adaptation, such as:
%                                 baseline values, sampling interval, z_vec, NP_vec, and NP_array.
%   H                           - Current measurement matrix.
%   R                           - Default measurement noise covariance matrix.
%   L1_c_over_n0_linear_estimates - Numeric array storing the estimated carrier-to-noise ratio (linear scale)
%                                 for each time step.
%   kf_type                     - char with the chosen KF type, which needs to be one of the following: {'standard', 'extended', 'unscented', 'cubature'}
%
% Outputs:
%   adapt_R                     - Adaptive measurement noise covariance computed for the current step.
%   extra_vars                  - Updated structure with any modified fields required for measurement adaptation.
%   L1_c_over_n0_linear_estimates - Updated estimates of the carrier-to-noise ratio (in linear scale),
%                                 including the latest estimate for the current time step.
% References:
% [1] E. Falletti, M. Pini and L. L. Presti, "Low Complexity 
%     Carrier-to-Noise Ratio Estimators for GNSS Digital Receivers," in 
%     IEEE Transactions on Aerospace and Electronic Systems, vol. 47, 
%     no. 1, pp. 420-437, January 2011, doi: 10.1109/TAES.2011.5705684. 
%     keywords: {Receivers;Signal to noise ratio;Estimation;Global 
%     Navigation Satellite Systems;Complexity theory;Correlators},
% [3] R. A. M. Lopes, F. Antreich, F. Fohlmeister, M. Kriegel and 
%     H. K. Kuga, "Ionospheric Scintillation Mitigation With Kalman 
%     PLLs Employing Radial Basis Function Networks", in IEEE Transactions 
%     on Aerospace and Electronic Systems, vol. 59, no. 5, pp. 6878-6893,
%     Oct. 2023, doi: 10.1109/TAES.2023.3281431
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    switch adaptive_config.measurement_cov_adapt_algorithm
        case 'none'
            adapt_R = R;
        case 'simplified'
            % NOTE: This method was inspired by equation 64 of [2]. It is
            % necessary to stablish a carrier to noise ratio baseline prior
            % to a scintillation event for utilizing this method.
            current_intensity = abs(received_signal(step-1, 1))^2;
            estimated_L1_c_over_n0_linear = current_intensity * extra_vars.baseline_L1_c_over_n0_linear;
            phase_noise_variance = (1 / (2 * estimated_L1_c_over_n0_linear * extra_vars.sampling_interval)) * ...
                                   (1 + (1 / (2 * estimated_L1_c_over_n0_linear * extra_vars.sampling_interval)));
            adapt_R = phase_noise_variance;
            L1_c_over_n0_linear_estimates(step, 1) = estimated_L1_c_over_n0_linear;
            extra_vars.latest_L1_c_over_n0_linear = estimated_L1_c_over_n0_linear;  % store for hard-limiting
        case 'nwpr'
            % NOTE: This estimator was based on the development proposed in
            % [1, Section III-E]
            NBP = (sum(real(extra_vars.z_vec)))^2 + (sum(imag(extra_vars.z_vec)))^2;
            WBP = sum(abs(extra_vars.z_vec).^2);
            NP = NBP / WBP;
            extra_vars.NP_vec = [NP; extra_vars.NP_vec(1:end-1)];
            mu_hat = (extra_vars.M_nwpr / extra_vars.N_nwpr) * sum(extra_vars.NP_vec);
            estimated_L1_c_over_n0_linear = (1 / extra_vars.sampling_interval) * (mu_hat - 1) / (extra_vars.M_nwpr - mu_hat);
            phase_noise_variance = (1 / (2 * estimated_L1_c_over_n0_linear * extra_vars.sampling_interval)) * ...
                                   (1 + (1 / (2 * estimated_L1_c_over_n0_linear * extra_vars.sampling_interval)));
            adapt_R = phase_noise_variance;
            % Update z_vec and NP_array.
            extra_vars.z_vec = [received_signal(step-1,1) * exp(-1j * H * extra_vars.x_hat_project_ahead); extra_vars.z_vec(1:end-1)];
            extra_vars.NP_array(step, 1) = NP;
            L1_c_over_n0_linear_estimates(step, 1) = estimated_L1_c_over_n0_linear;
            extra_vars.latest_L1_c_over_n0_linear = estimated_L1_c_over_n0_linear;
        otherwise
            adapt_R = R;
            %TODO: Make the adaptive to the other kf variant types
    end
end

function [K, innovation, x_hat_update, P_hat_update] = standard_kf_update(...
    step, received_signal, extra_vars, H, adapt_R, R)
% standard_kf_update
%
% Computes the Kalman gain, innovation, and updated state and error covariance
% estimates for the standard Kalman filter.
%
% Syntax:
%   [K, innovation, extra_vars, x_hat_update, P_hat_update] = standard_kf_update(...
%         step, received_signal, extra_vars, H, adapt_R, R)
%
% Description:
%   For the standard Kalman filter, the measurement update is performed using a linear
%   innovation. This function computes the estimated signal as:
%       estimated_signal = exp(1j * H * extra_vars.x_hat_project_ahead)
%   and then calculates the innovation as the angle difference between the received
%   signal (from the previous time step) and the conjugate of the estimated signal.
%   The Kalman gain is then computed using the current error covariance and the adaptive
%   measurement noise covariance (adapt_R). Finally, the function updates the state and
%   covariance estimates using the standard KF update equations.
%
% Inputs:
%   step            - Scalar indicating the current time step.
%   received_signal - Numeric 2D array representing the received signal.
%   extra_vars      - Structure containing additional filtering variables, including:
%                     x_hat_project_ahead: Current projected state estimate.
%                     P_hat_project_ahead: Current error covariance estimate.
%   H               - Measurement matrix.
%   adapt_R         - Adaptive measurement noise covariance computed for the current step.
%   R               - Default measurement noise covariance matrix.
%
% Outputs:
%   K               - Computed Kalman gain for the standard Kalman filter.
%   innovation      - Innovation value computed as the angle between the received signal
%                     (at step-1) and the estimated signal.
%   extra_vars      - Unmodified structure of additional filtering variables (passed through).
%   x_hat_update    - Updated state estimate after applying the Kalman gain.
%   P_hat_update    - Updated error covariance estimate after the measurement update.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    if extra_vars.is_geometry_aided && size(H, 1) == 2
        if isscalar(adapt_R)
            if isempty(extra_vars.geometry_noise_variance)
                error('get_kalman_pll_estimates:missing_geometry_noise_variance', ...
                    'Geometry-aided KF requires geometry_noise_variance in the config.');
            end
            adapt_R = diag([adapt_R, extra_vars.geometry_noise_variance]);
        end
        H_total = H(1, :);
        H_geometry = H(2, :);
        estimated_signal = exp(1j * H_total * extra_vars.x_hat_project_ahead);
        innovation_total = angle(received_signal(step-1, 1) * conj(estimated_signal));
        innovation_geometry = extra_vars.geometry_measurement(step-1, 1) - (H_geometry * extra_vars.x_hat_project_ahead);
        innovation = [innovation_total; innovation_geometry];
    else
        % Compute the estimated signal using the linear measurement model.
        estimated_signal = exp(1j * H * extra_vars.x_hat_project_ahead);
        
        % Compute innovation as the difference in angle between the received signal
        % (from the previous time step) and the estimated signal.
        innovation = angle(received_signal(step-1, 1) * conj(estimated_signal));
    end
    
    % Compute the Kalman gain.
    K = extra_vars.P_hat_project_ahead * H.' * ((H * extra_vars.P_hat_project_ahead * H.' + adapt_R) \ eye(size(R, 1)));
    
    % Update the state estimate using the computed gain and innovation.
    x_hat_update = extra_vars.x_hat_project_ahead + K * innovation;
    
    % Update the error covariance estimate.
    P_hat_update = extra_vars.P_hat_project_ahead - K * H * extra_vars.P_hat_project_ahead;
end

function [K, innovation, x_hat_update, P_hat_update] = extended_kf_update(...
    step, received_signal, Hj_handle, extra_vars, adapt_R, R, psi_tppsm)
% extended_kf_update
%
% Computes the Kalman gain and the innovation term for the extended Kalman
% filter. This function uses the provided Jacobian function handle 
% (Hj_handle) to update the linearized measurement matrix based on the 
% current estimated state, then calculates the innovation by comparing 
% the received signal with the estimated signal, and finally computes the
% Kalman gain.
%
% Syntax:
%   [K, innovation, H, extra_vars] = extended_kf_update(...
%         step, received_signal, Hj_handle, extra_vars, adapt_R, R)
%
% Description:
%   For the extended KF, the measurement matrix H is updated by computing 
%   the Jacobian at the current projected state (stored in 
%   extra_vars.x_hat_project_ahead) using the measurement function. The 
%   function then forms the innovation vector by subtracting the real and 
%   imaginary components of the estimated signal from those of the received 
%   signal. Finally, the Kalman gain is computed using the updated 
%   measurement matrix H, the current error covariance (stored in 
%   extra_vars.P_hat_project_ahead), and the adaptive measurement
%   noise covariance adapt_R.
%
% Inputs:
%   step           - Scalar indicating the current iteration (time step).
%   received_signal- Numeric 2D array representing the received signal.
%   Hj_handle      - Function handle to compute the Jacobian measurement matrix for the extended KF.
%   extra_vars     - Structure containing additional filtering variables, including:
%                    x_hat_project_ahead: the current projected state estimate,
%                    P_hat_project_ahead: the current error covariance estimate.
%   adapt_R        - Adaptive measurement noise covariance computed for the current step.
%   R              - Default measurement noise covariance matrix.
%
% Outputs:
%   K              - Computed Kalman gain for the extended filter.
%   innovation     - Innovation vector (difference between actual and estimated measurements).
%   H              - Updated measurement matrix computed using the Jacobian function.
%   extra_vars     - Input extra_vars structure (returned unchanged in this function).
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com
    
    external_amplitude = abs(psi_tppsm(step - 1, 1));
    H = Hj_handle(external_amplitude, extra_vars.x_hat_project_ahead);
    
    estimated_signal = exp(1j * extra_vars.x_hat_project_ahead(1));
    innovation = [real(received_signal(step-1, 1)); imag(received_signal(step-1, 1))] - ...
                 [real(estimated_signal); imag(estimated_signal)];
    
    % Compute Kalman gain.
    K = extra_vars.P_hat_project_ahead * H.' * ((H * extra_vars.P_hat_project_ahead * H.' + adapt_R) \ eye(size(R, 1)));

    % Update the state estimates.
    x_hat_update = extra_vars.x_hat_project_ahead + K * innovation;
    
    % Update error covariance.
    P_hat_update = extra_vars.P_hat_project_ahead - K * H * extra_vars.P_hat_project_ahead;
end

function [K, innovation, x_hat_update, P_hat_update] = unscented_kf_update(...
    step, received_signal, extra_vars, adapt_R, ut_params, psi_tppsm)
% unscented_kf_update
%
% Performs the UKF measurement update by:
%   - Generating sigma points from the projected state.
%   - Mapping the sigma points through a nonlinear measurement function:
%         h(a, x) = a * [cos(x(1)); sin(x(1))]
%     where 'a' is the external amplitude (from the received signal) and x(1)
%     is the phase component of the sigma point.
%   - Computing the predicted measurement mean, measurement covariance, and
%     state-measurement cross-covariance.
%   - Calculating the Kalman gain and updating the state and covariance estimates.
%
% Equations (based on [1], Equations 7.5.10 to 7.5.17):
%   (7.5.10) Sigma point generation:
%         χ₀ = x_pred;
%         χᵢ = x_pred + (√((n+λ)P))ᵢ,    i = 1,...,n;
%         χ₍ₙ₊ᵢ₎ = x̂_pred - (√((n+λ)P))ᵢ, i = 1,...,n.
%
%   (7.5.11) Nonlinear measurement mapping:
%         zᵢ = h(a, χᵢ) = a * [cos(χᵢ(1)); sin(χᵢ(1))],
%         where 'a' is the external amplitude.
%
%   (7.5.12) Predicted measurement mean:
%         z_pred = ω₀^m * z₀ + Σ_{i=1}^{2n} ωᵢ^m * zᵢ.
%
%   (7.5.13) Measurement covariance:
%         P_zz = ω₀^c (z₀ - z_pred)(z₀ - z_pred)' +
%                Σ_{i=1}^{2n} ωᵢ^c (zᵢ - z_pred)(zᵢ - z_pred)' + R.
%
%   (7.5.14) Cross covariance:
%         P_xz = ω₀^c (χ₀ - x̂_pred)(z₀ - z_pred)' +
%                Σ_{i=1}^{2n} ωᵢ^c (χᵢ - x̂_pred)(zᵢ - z_pred)'.
%
%   (7.5.15) Kalman gain:
%         K = P_xz * inv(P_zz)  (implemented as K = P_xz*(P_zz\eye(size(P_zz,1)))).
%
%   (7.5.16) State update:
%         x̂_update = x̂_pred + K*(z_actual - z_pred).
%
%   (7.5.17) Covariance update:
%         P_update = P_pred - K * P_zz * K'.
%
% Inputs:
%   step            - Current time step (scalar).
%   received_signal - Numeric 2D array representing the received signal.
%   extra_vars      - Struct with:
%                     x_hat_project_ahead: projected state estimate (n x 1)
%                     P_hat_project_ahead: projected error covariance (n x n)
%   H               - (Not used; provided for interface compatibility.)
%   adapt_R         - Adaptive measurement noise covariance.
%   R               - Default measurement noise covariance.
%   ut_params       - Struct with unscented transform parameters:
%                       states_amount: number of states (n)
%                       sigma_points_amount: 2*n + 1
%                       lambda: scaling parameter
%                       omega_mean_0: weight for the 0th sigma point (mean)
%                       omega_covariance_0: weight for the 0th sigma point (covariance)
%                       omega_mean_covariance_general: weight for the remaining sigma points
%
% Outputs:
%   K               - Kalman gain matrix.
%   innovation      - Measurement innovation (z_actual - z_pred).
%   x_hat_update    - Updated state estimate.
%   P_hat_update    - Updated error covariance.
%
% Reference:
%   [1] Brown, R.G. & Hwang, P.Y.C., “Introduction to Random Signals and 
%       Applied Kalman Filtering: With MATLAB® Exercises, Fourth Edition”
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    % Generate sigma points for the projected state, based on [1, equation 
    % 7.5.10]
    n = ut_params.states_amount;
    num_sigma = ut_params.sigma_points_amount;  % 2*n + 1
    X = zeros(n, num_sigma);
    sqrt_mat = sqrtm((n + ut_params.lambda) * extra_vars.P_hat_project_ahead);
    X(:,1) = extra_vars.x_hat_project_ahead;
    for i = 2 : n+1
        X(:,i) = extra_vars.x_hat_project_ahead + sqrt_mat(:, i-1);
    end
    for i = n+2 : num_sigma
        X(:,i) = extra_vars.x_hat_project_ahead - sqrt_mat(:, i - n - 1);
    end

    % Map sigma points through the nonlinear measurement function, based on
    % [1, equation 7.5.11].
    switch extra_vars.measurement_model
        case 'complex_field'
            angles = X(extra_vars.idx.phase, :);
            field_real = X(extra_vars.idx.field_real, :);
            field_imag = X(extra_vars.idx.field_imag, :);
            Z = [field_real .* cos(angles) - field_imag .* sin(angles); ...
                 field_real .* sin(angles) + field_imag .* cos(angles)];
        case 'external_amplitude_phase'
            if isempty(psi_tppsm)
                error('get_kalman_pll_estimates:missing_external_amplitude', ...
                    'The unscented KF without complex-field augmentation requires psi_tppsm to provide the external amplitude.');
            end
            external_amp = abs(psi_tppsm(step - 1, 1));
            angles = X(1, :);  % use only the first element of each sigma point
            Z = external_amp * [cos(angles); sin(angles)];  % 2 x num_sigma
        otherwise
            error('get_kalman_pll_estimates:invalid_measurement_model', ...
                'Unsupported UKF measurement model: %s.', extra_vars.measurement_model);
    end

    % Compute predicted measurement mean [1, equation 7.5.12]
    z_pred = ut_params.omega_mean_0 * Z(:,1) + ut_params.omega_mean_covariance_general * sum(Z(:,2:end), 2);

    % Compute measurement covariance P_zz [1, equation 7.5.13]
    meas_diffs = Z - repmat(z_pred, 1, num_sigma);
    P_zz = ut_params.omega_covariance_0 * (meas_diffs(:,1) * meas_diffs(:,1).') + ...
           ut_params.omega_mean_covariance_general * (meas_diffs(:,2:end) * meas_diffs(:,2:end).');
    P_zz = P_zz + adapt_R;
    
    % Compute cross-covariance P_xz [1, equation 7.5.14]
    state_diffs = X - repmat(extra_vars.x_hat_project_ahead, 1, num_sigma);
    P_xz = ut_params.omega_covariance_0 * (state_diffs(:,1) * meas_diffs(:,1).') + ...
           ut_params.omega_mean_covariance_general * (state_diffs(:,2:end) * meas_diffs(:,2:end).');
    
    % Compute Kalman gain [1, equation 7.5.15]
    K = P_xz * (P_zz \ eye(size(P_zz, 1)));
    
    % Separate the actual received signal measurement in real and
    % imaginary components.
    z_actual = [real(received_signal(step-1, 1)); imag(received_signal(step-1, 1))];
    
    % Compute innovation
    innovation = z_actual - z_pred;
    
    % Update state estimates [1, equation 7.5.16]
    x_hat_update = extra_vars.x_hat_project_ahead + K * innovation;
    
    %Update state covariance estimates [1, equation 7.5.17] 
    P_hat_update = extra_vars.P_hat_project_ahead - K * P_zz * K.';
end

function extra_vars = update_state_covariance(step, received_signal, adaptive_config, extra_vars, Q, H, innovation, K)
% update_state_covariance
%
% Adapts the state covariance matrix using the 'matching' algorithm based on the method
% specified in adaptive_config.states_cov_adapt_algorithm_params ('IAE' or 'RAE').
%
% Syntax:
%   extra_vars = update_state_covariance(step, received_signal, adaptive_config, ...
%                                        extra_vars, Q, H, innovation, K)
%
% Description:
%   This function updates the state covariance matrix by first computing an error term
%   (phi for the IAE method or mu for the RAE method) at the previous time step using the
%   received signal and the current state estimate. The error is stored in a sliding window
%   (extra_vars.states_error_mem). When sufficient measurements have been collected 
%   (i.e., when the current step is greater than or equal to the specified window size),
%   the function computes the updated covariance matrix and stores it in extra_vars.updated_Q.
%   Otherwise, the input covariance Q is retained.
%
% Inputs:
%   step            - Scalar indicating the current time step.
%   received_signal - Numeric array representing the received signal.
%   adaptive_config - Struct containing adaptive configuration settings, including the field
%                     states_cov_adapt_algorithm_params (with fields 'method' and 'window_size').
%   extra_vars      - Struct containing additional filtering variables, including:
%                        x_hat_project_ahead: the current projected state estimate.
%                        states_error_mem: the sliding memory for state error values.
%   Q               - Current process noise covariance matrix.
%   H               - Current measurement matrix.
%   innovation      - Innovation vector computed for the current time step.
%   K               - Kalman gain matrix.
%
% Outputs:
%   extra_vars      - Updated extra_vars structure with the field updated_Q containing the new
%                     state covariance if the window size condition is met; otherwise, Q is preserved.
%
% Reference:
%  [1] Chen Y-W, Tu K-M. Robust self-adaptive Kalman filter with 
%      application in target tracking. Measurement and Control. 
%      2022;55(9-10):935-944. doi:10.1177/00202940221083548
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com
    
    % NOTE: The adaptive algorithm used here is known as Covariance
    % Matching. Please, refer to [1] for more details. 
    %
    % In the case considered in this work, the adaptation of the 
    % measurement covariance matrix is only dependent on the measured 
    % carrier to noise ratio, to avoid filter instabilities that may 
    % happen when one tries to estimate both measurement and state 
    % covariances simulteanously using solely the covariance matching 
    % method.
    if strcmpi(adaptive_config.states_cov_adapt_algorithm_params.method, 'IAE')
        phi = angle(received_signal(step-1,1) * exp(-1j * (H * extra_vars.x_hat_project_ahead)));
        extra_vars.states_error_mem = [phi; extra_vars.states_error_mem(1:end-1)];
        if step >= adaptive_config.states_cov_adapt_algorithm_params.window_size
            extra_vars.updated_Q = K * (1 / adaptive_config.states_cov_adapt_algorithm_params.window_size) * ...
                (extra_vars.states_error_mem.' * extra_vars.states_error_mem) * K.';
        else
            extra_vars.updated_Q = Q;
        end
    elseif strcmpi(adaptive_config.states_cov_adapt_algorithm_params.method, 'RAE')
        mu = angle(received_signal(step-1,1) * exp(-1j * (H * (extra_vars.x_hat_project_ahead + K * innovation))));
        extra_vars.states_error_mem = [mu; extra_vars.states_error_mem(1:end-1)];
        if step >= adaptive_config.states_cov_adapt_algorithm_params.window_size
            extra_vars.updated_Q = K * (1 / adaptive_config.states_cov_adapt_algorithm_params.window_size) * ...
                (extra_vars.states_error_mem.' * extra_vars.states_error_mem) * K.';
        else
            extra_vars.updated_Q = Q;
        end
    end
end

%% Helper Functions
function validate_received_signal(received_signal)
% validate_received_signal
%
% Validates that the input received_signal is a nonempty numeric 2D array.
%
% Syntax:
%   validate_received_signal(received_signal)
%
% Description:
%   This function checks that the input received_signal meets the following criteria:
%     - It is a numeric array.
%     - It is nonempty.
%     - It is two-dimensional.
%
% Inputs:
%   received_signal - Numeric 2D array representing the received signal.
%
% Outputs:
%   None. An error is thrown if the validation fails.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    validateattributes(received_signal, {'numeric'}, {'nonempty', '2d'}, mfilename, 'received_signal');
end

function validate_initial_estimates(initial_estimates)
% validate_initial_estimates
%
% Validates that the input initial_estimates is a nonempty struct with the required fields.
%
% Syntax:
%   validate_initial_estimates(initial_estimates)
%
% Description:
%   This function checks that the initial_estimates struct:
%     - Is nonempty and a struct.
%     - Contains the fields 'x_hat_init' and 'P_hat_init'.
%     - The field 'x_hat_init' is a nonempty numeric vector.
%     - The field 'P_hat_init' is a nonempty numeric square matrix whose size corresponds
%       to the number of elements in x_hat_init.
%
% Inputs:
%   initial_estimates - Struct with fields:
%                       x_hat_init: Numeric column vector representing the initial state estimate.
%                       P_hat_init: Numeric square matrix representing the initial error covariance.
%
% Outputs:
%   None. An error is thrown if any of the validation checks fail.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    validateattributes(initial_estimates, {'struct'}, {'nonempty'}, mfilename, 'initial_estimates');
    if ~isfield(initial_estimates, 'x_hat_init') || ~isfield(initial_estimates, 'P_hat_init')
        error('get_kalman_pll_estimates:missing_field', 'initial_estimates must have fields "x_hat_init" and "P_hat_init".');
    end
    validateattributes(initial_estimates.x_hat_init, {'numeric'}, {'nonempty', 'vector'}, mfilename, 'initial_estimates.x_hat_init');
    n = numel(initial_estimates.x_hat_init);
    validateattributes(initial_estimates.P_hat_init, {'numeric'}, {'nonempty', '2d', 'size', [n n]}, mfilename, 'initial_estimates.P_hat_init');
end

function validate_training_scint_model(training_scint_model, kalman_pll_config)
% validate_training_scint_model
%
% Validates that the input training_scint_model is a valid character vector or string
% that corresponds to a field in the kalman_pll_config structure.
%
% Syntax:
%   validate_training_scint_model(training_scint_model, kalman_pll_config)
%
% Description:
%   This function checks that:
%     - The training_scint_model is either a char or a string.
%     - Its value is one of the allowed options: 'CSM', 'TPPSM', or 'none'.
%     - The kalman_pll_config structure contains a field with the name of training_scint_model.
%
% Inputs:
%   training_scint_model - A char or string indicating the scintillation model used to train the AR model.
%                          Allowed values: 'CSM', 'TPPSM', 'none'.
%   kalman_pll_config    - Struct containing the Kalman filter configuration. This struct is expected
%                          to have a field corresponding to training_scint_model.
%
% Outputs:
%   None. An error is thrown if any of the validation checks fail.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    if ~(ischar(training_scint_model) || isstring(training_scint_model))
        error('get_kalman_pll_estimates:invalid_type', 'training_scint_model must be a char or string.');
    end
    training_scint_model = char(training_scint_model);
    valid_models = {'CSM', 'TPPSM', 'none'};
    if ~any(strcmpi(training_scint_model, valid_models))
        error('get_kalman_pll_estimates:invalid_training_model', 'training_scint_model must be one of: %s.', strjoin(valid_models, ', '));
    end
    if ~isfield(kalman_pll_config, training_scint_model)
        error('get_kalman_pll_estimates:invalid_training_model', 'kalman_pll_config does not have the field for training_scint_model: %s', training_scint_model);
    end
end

function validate_adaptive_config(adaptive_config)
% validate_adaptive_config
%
% Validates the adaptive configuration struct used by the Kalman filter.
%
% Syntax:
%   validate_adaptive_config(adaptive_config)
%
% Description:
%   This function verifies that the input adaptive_config structure contains all the
%   required fields for adaptive configuration. It checks for the existence and validity
%   of the following:
%
%     1. The main fields: 'measurement_cov_adapt_algorithm', 'states_cov_adapt_algorithm',
%        'sampling_interval', and 'hard_limited'.
%
%     2. The 'sampling_interval' field is a positive scalar.
%
%     3. The 'hard_limited' field is a nonempty struct that contains the field 'is_used'
%        (a scalar logical or numeric value), and if hard limiting is enabled, it must also
%        contain the field 'L1_C_over_N0_dBHz_threshold' (a numeric scalar).
%
%     4. The measurement covariance adaptation:
%        - Valid algorithms are 'none', 'simplified', or 'nwpr'.
%        - When an algorithm other than 'none' is specified, the struct must include the field
%          'measurement_cov_adapt_algorithm_params' containing the required parameters. For
%          'simplified', the required field is 'L1_C_over_N0_dBHz'; for 'nwpr', the required
%          fields are 'N_nwpr' and 'M_nwpr'.
%
%     5. The states covariance adaptation:
%        - Valid algorithms are 'none' or 'matching'.
%        - If 'matching' is specified, the struct must include the field
%          'states_cov_adapt_algorithm_params' containing the required fields 'method' (which
%          must be either 'IAE' or 'RAE') and 'window_size'.
%
% Inputs:
%   adaptive_config - Struct containing the adaptive configuration options for the Kalman filter.
%                     This struct must include fields for measurement and state covariance adaptation,
%                     as well as the global sampling interval and hard limiting settings.
%
% Outputs:
%   None. An error is thrown if any validation check fails.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    validateattributes(adaptive_config, {'struct'}, {'nonempty'}, mfilename, 'adaptive_config');

    % Required main fields
    required_main_fields = {'measurement_cov_adapt_algorithm', 'states_cov_adapt_algorithm', 'sampling_interval', 'hard_limited'};
    for i = 1:numel(required_main_fields)
        if ~isfield(adaptive_config, required_main_fields{i})
            error('validate_adaptive_config:missing_field', ...
                  'adaptive_config must contain the field "%s".', required_main_fields{i});
        end
    end

    % Validate sampling_interval
    validateattributes(adaptive_config.sampling_interval, {'numeric'}, {'scalar', 'positive'}, ...
                       mfilename, 'adaptive_config.sampling_interval');

    % Validate hard_limited
    hl = adaptive_config.hard_limited;
    validateattributes(hl, {'struct'}, {'nonempty'}, mfilename, 'adaptive_config.hard_limited');
    if ~isfield(hl,'is_used')
        error('validate_adaptive_config:missing_hard_limited_field', ...
                  'hard_limited must have field "%s".', 'is_used');
    end
    validateattributes(hl.is_used, {'logical', 'numeric'}, {'scalar'}, mfilename, 'hard_limited.is_used');
    if hl.is_used
        if ~isfield(hl, 'L1_C_over_N0_dBHz_threshold')
            error('validate_adaptive_config:missing_hard_limited_field', ...
                  'hard_limited must have field "%s".', 'L1_C_over_N0_dBHz_threshold');
        end
        validateattributes(hl.L1_C_over_N0_dBHz_threshold, {'numeric'}, {'scalar'}, ...
                       mfilename, 'hard_limited.L1_C_over_N0_dBHz_threshold');
    end
    
    % Validate measurement_cov_adapt_algorithm
    valid_meas_algos = {'none', 'simplified', 'nwpr'};
    algo = adaptive_config.measurement_cov_adapt_algorithm;
    assert(any(strcmpi(algo, valid_meas_algos)), ...
        'validate_adaptive_config:invalid_algorithm', ...
        'Invalid measurement_cov_adapt_algorithm. Allowed values are: %s', strjoin(valid_meas_algos, ', '));

    if hl.is_used && strcmpi(algo, 'none')
        error('validate_adaptive_config:hard_limited_without_adaptive', ...
              ['The Hard limited option should only be configured to be used ' ...
              'when a measurement covariance algorithm is also used.'])
    end
    if ~strcmpi(algo, 'none')
        if ~isfield(adaptive_config, 'measurement_cov_adapt_algorithm_params')
            error('validate_adaptive_config:missing_field', ...
                  'Field "measurement_cov_adapt_algorithm_params" is required for algorithm "%s".', algo);
        end
        p = adaptive_config.measurement_cov_adapt_algorithm_params;
        validateattributes(p, {'struct'}, {'nonempty'}, mfilename, 'measurement_cov_adapt_algorithm_params');
        % Shared required fields
        if strcmpi(algo, 'simplified')
            check_fields(p, {'L1_C_over_N0_dBHz'}, 'measurement_cov_adapt_algorithm_params');
        elseif strcmpi(algo, 'nwpr')
            fields = {'N_nwpr','M_nwpr'};
            check_fields(p, fields, 'measurement_cov_adapt_algorithm_params');
        end
    end

    % Validate states_cov_adapt_algorithm
    valid_states_algos = {'none', 'matching'};
    s_algo = adaptive_config.states_cov_adapt_algorithm;
    assert(any(strcmpi(s_algo, valid_states_algos)), ...
        'validate_adaptive_config:invalid_states_algorithm', ...
        'Invalid states_cov_adapt_algorithm. Allowed values are: %s', strjoin(valid_states_algos, ', '));

    if strcmpi(s_algo, 'matching')
        if ~isfield(adaptive_config, 'states_cov_adapt_algorithm_params')
            error('validate_adaptive_config:missing_field', ...
                  'Field "states_cov_adapt_algorithm_params" is required for algorithm "%s".', s_algo);
        end
        p = adaptive_config.states_cov_adapt_algorithm_params;
        required_fields = {'method', 'window_size'};
        check_fields(p, required_fields, 'states_cov_adapt_algorithm_params');
        if ~any(strcmpi(p.method, {'IAE', 'RAE'}))
            error('validate_adaptive_config:invalid_method', ...
                  'Invalid method in states_cov_adapt_algorithm_params. Must be either "IAE" or "RAE".');
        end
    end
end

function check_fields(structure, field_list, context)
% check_fields
%
% Checks that a given structure contains all required fields.
%
% Syntax:
%   check_fields(structure, field_list, context)
%
% Description:
%   This function iterates over the list of required field names provided in field_list and
%   verifies that each field exists in the given structure. If any field is missing, an error
%   is thrown with a message indicating which field is missing in the specified context.
%
% Inputs:
%   structure  - A struct that is expected to contain specific fields.
%   field_list - Cell array of strings containing the names of the required fields.
%   context    - A string indicating the context (e.g., a substructure name) in which the
%                fields are required.
%
% Outputs:
%   None. An error is thrown if any field in field_list is not present in structure.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    for i = 1:numel(field_list)
        if ~isfield(structure, field_list{i})
            error('validate_adaptive_config:missing_field', ...
                  '%s must contain the field "%s".', context, field_list{i});
        end
    end
end

function validate_online_learning_cfg(online_mdl_learning_cfg, kalman_pll_config, training_scint_model, kf_type)
% validate_online_learning_cfg
%
% Validates the online model learning configuration used for the augmentation model
% in the Kalman PLL estimator. This function ensures that the online_mdl_learning_cfg
% struct contains the required fields and that they are consistent with the offline
% augmentation model specified in kalman_pll_config for the given training_scint_model.
%
% Syntax:
%   validate_online_learning_cfg(online_mdl_learning_cfg, kalman_pll_config, training_scint_model)
%
% Description:
%   This function first validates that online_mdl_learning_cfg is a nonempty struct and 
%   that it contains the field 'is_online'. If online model learning is enabled 
%   (i.e., online_mdl_learning_cfg.is_online is true), it further verifies that the 
%   kalman_pll_config structure contains an 'augmentation_model_initializer' field for the 
%   specified training_scint_model. It then checks that the augmentation model initializer 
%   contains an 'id' field and that this id is of type char or string.
%
%   Depending on the offline augmentation model id (e.g., 'arfit', 'aryule', 'rbf', 'kinematic', or 'none'),
%   the function validates additional fields in online_mdl_learning_cfg, such as:
%     - For AR models ('arfit' or 'aryule'): The 'learning_method' field must be present and,
%       if the learning method is 'sliding_window' or 'block_window', a numeric positive scalar
%       'window_size' must also be provided.
%     - Other augmentation model types (e.g., 'rbf' or 'kinematic') are currently not supported.
%
% Inputs:
%   online_mdl_learning_cfg - Struct with settings for online model learning for the augmentation model.
%                             It must have the field 'is_online' (boolean) and, when online learning is enabled,
%                             it may require additional fields depending on the offline augmentation model id.
%   kalman_pll_config       - Struct containing the Kalman filter configuration, which must include the
%                             'augmentation_model_initializer' field for the given training_scint_model.
%   training_scint_model    - A string or char specifying the scintillation model used for training.
%   kf_type                 - A KF type ({'standard', 'extended', 'unscented', 'cubature'})
%
% Outputs:
%   None. An error is thrown if any validation check fails.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    validateattributes(online_mdl_learning_cfg, {'struct'}, {'nonempty'}, mfilename, 'online_mdl_learning_cfg');
    
    if ~isfield(online_mdl_learning_cfg, 'is_online')
        error('get_kalman_pll_estimates:missing_field', ...
              'online_mdl_learning_cfg must have a field "is_online" that indicates if online model learning is enabled.');
    end
    validateattributes(online_mdl_learning_cfg.is_online, {'logical'}, {}, mfilename, 'online_mdl_learning_cfg.is_online');
    
    if online_mdl_learning_cfg.is_online
        % Get the offline augmentation model identifier.
        if ~isfield(kalman_pll_config.(training_scint_model).(kf_type), 'augmentation_model_initializer')
            error('get_kalman_pll_estimates:missing_field', ...
                  'kalman_pll_config must have a field "augmentation_model_initializer" specifying the offline augmentation model.');
        end
        if ~isfield(kalman_pll_config.(training_scint_model).(kf_type).augmentation_model_initializer, 'id')
            error('get_kalman_pll_estimates:missing_field', ...
                  'kalman_pll_config.augmentation_model_initializer must contain the field "id" that identifies the augmentation model used in offline training.');
        end
        offline_id = kalman_pll_config.(training_scint_model).(kf_type).augmentation_model_initializer.id;
        if ~(ischar(offline_id) || isstring(offline_id))
            error('get_kalman_pll_estimates:invalid_type', ...
                  'The field kalman_pll_config.augmentation_model_initializer.id must be a char or string.');
        end
        offline_id = char(offline_id);
        
        switch lower(offline_id)
            case {'arfit', 'aryule'}
                % For AR models, require the fields directly in online_mdl_learning_cfg.
                if ~isfield(online_mdl_learning_cfg, 'learning_method')
                    error('get_kalman_pll_estimates:missing_field', ...
                          'For offline augmentation model id "%s", online_mdl_learning_cfg must have a field "learning_method".', offline_id);
                end
                validateattributes(online_mdl_learning_cfg.learning_method, {'char','string'}, {'nonempty'}, mfilename, 'online_mdl_learning_cfg.learning_method');
                method = lower(char(online_mdl_learning_cfg.learning_method));
                if ismember(method, {'sliding_window', 'block_window'})
                    if ~isfield(online_mdl_learning_cfg, 'window_size')
                        error('get_kalman_pll_estimates:missing_field', ...
                              'For learning method "%s" with offline model id "%s", online_mdl_learning_cfg must include a "window_size" field.', method, offline_id);
                    end
                    validateattributes(online_mdl_learning_cfg.window_size, {'numeric'}, {'scalar','positive'}, mfilename, 'online_mdl_learning_cfg.window_size');
                end
                if ismember(method, 'kalman')
                    error('get_kalman_pll_estimates:NotSupportedLearningMethod', 'Adapting the AR coefficients using a Kalman filter is not yet supported.')
                end
            case 'arfit-complex-field'
                error('get_kalman_pll_estimates:unsupported_online_complex_field', ...
                    'Online learning is not supported for the arfit-complex-field augmentation model.');
            case 'rbf'
                error('get_kalman_pll_estimates:NotSupported', 'RBF online learning is not yet supported.');
            case 'kinematic'
                error('Online model estimation for ''second_wiener_mdl'' is not yet supported.');
            case 'none'
                % Does nothing
            otherwise
                error('get_kalman_pll_estimates:invalid_augmentation_id', ...
                      'Unsupported offline augmentation model id "%s" found in kalman_pll_config.augmentation_model_initializer.id', offline_id);
        end
    end
end

function validate_kf_type(kf_type)
% validate_kf_type
%
% Validates that the kf_type input is one of the allowed Kalman filter types.
%
% Syntax:
%   validate_kf_type(kf_type)
%
% Description:
%   This function checks that the input kf_type is either a char or a string,
%   and that it matches one of the following allowed values: 'standard', 'extended',
%   'unscented', or 'cubature'. An error is thrown if the validation fails.
%
% Inputs:
%   kf_type - A char or string specifying the Kalman filter type. Allowed values:
%             'standard', 'extended', 'unscented', 'cubature'.
%
% Outputs:
%   None. An error is raised if kf_type does not meet the specified criteria.
%
% Example:
%   validate_kf_type('extended');
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com
    
    % Check if kf_type is a char or string.
    if ~(ischar(kf_type) || isstring(kf_type))
        error('get_kalman_pll_estimates:invalid_kf_type', ...
            'kf_type must be a char or string.');
    end

    % Convert the kf_type to char for uniformity.
    kf_type = char(kf_type);
    
    % Define the allowed types.
    allowed_types = {'standard', 'extended', 'unscented', 'cubature'};
    
    % Validate that the input is one of the allowed types.
    if ~any(strcmpi(kf_type, allowed_types))
        error('get_kalman_pll_estimates:invalid_kf_type', ...
            'kf_type must be one of the following: %s.', strjoin(allowed_types, ', '));
    end
end
