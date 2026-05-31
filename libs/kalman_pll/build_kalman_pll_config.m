function kalman_pll_config = build_kalman_pll_config(general_config, kalman_pll_config)
% build_kalman_pll_config
%
% Summary:
%   Configures and constructs the Kalman filter and VAR (or alternative
%   augmentation model) state-space matrices based on the provided configuration.
%   The function computes line-of-sight (LOS) dynamics using a discrete Wiener model,
%   preprocesses training data, and—depending on the selected augmentation model and
%   KF variant—computes the necessary matrices for the state-space filter.
%
% Syntax:
%   kalman_pll_config = build_kalman_pll_config(general_config, kalman_pll_config)
%
% Description:
%   This function derives the Kalman filter and VAR/augmentation model settings 
%   from user-specified configuration parameters. It performs the following steps:
%
%     1. Computes the LOS dynamics matrices (F_los and Q_los) from the discrete
%        Wiener model specified in general_config.discrete_wiener_model_config.
%
%     2. Preprocesses the training data from general_config.scintillation_training_data_config.
%
%     3. Computes augmentation-related matrices (or flags none) by calling
%        compute_augmentation.
%
%     4. Depending on the chosen KF variant (general_config.kf_type), constructs the
%        full state-space matrices. For now, only the 'standard' KF variant is implemented.
%
%     5. Stores the computed matrices and augmentation model initializer in a subfield of
%        the output structure, keyed by the scintillation model.
%
% Inputs:
%   general_config - Struct containing configuration settings, including:
%       - discrete_wiener_model_config: Cell array {L, M, sampling_interval, sigma, delta}.
%       - scintillation_training_data_config: Struct with training data and scintillation model info.
%       - C_over_N0_array_dBHz: Numeric vector of average C/N0 values in dBHz.
%       - augmentation_model_initializer: Struct with fields:
%           * id: Augmentation model identifier ('arfit', 'aryule', 'kinematic', 'arima', 'rbf', 'none').
%           * model_params: Parameters specific to the augmentation method.
%       - kf_type: String indicating the chosen KF variant ('standard', 'extended',
%                  'unscented', or 'cubature').
%       - Other parameters as required.
%
%   kalman_pll_config - Struct holding or updating previous Kalman PLL settings.
%
% Outputs:
%   kalman_pll_config - Struct including a substructure (named after the
%                       scintillation model) that comprises:
%           F   : Full state transition matrix.
%           Q   : Full process noise covariance matrix.
%           H   : Measurement matrix.
%           R   : Measurement noise covariance matrix.
%           W   : Additional matrix (if applicable).
%           augmentation_model_initializer : Original augmentation initializer structure.
%
% Example:
%   kalman_pll_config = build_kalman_pll_config(general_config, kalman_pll_config);
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    % 1. Compute the LOS dynamics matrices.
    [F_los, Q_los] = get_discrete_wiener_model(general_config.discrete_wiener_model_config{:});
    
    % 2. Preprocess training data.
    training_data = preprocess_training_data( ...
        general_config.scintillation_training_data_config, ...
        general_config.augmentation_model_initializer.id);
    sampling_interval = general_config.discrete_wiener_model_config{3};
    
    % 3. Compute augmentation data using the new helper.
    aug_data = get_augmentation_model(training_data, general_config, sampling_interval);
    
    % 4. Select the KF variant and build full KF matrices.
    kf_type = lower(string(general_config.kf_type));
    switch kf_type
        case 'standard'
            [F, Q, H, R, W] = build_standard_kf(F_los, Q_los, aug_data, general_config, sampling_interval);
            Hj_handle = [];
        case 'extended'
            [F, Q, Hj_handle, R] = build_extended_kf(F_los, Q_los, aug_data, general_config, sampling_interval);
            W = zeros(size(F_los,1), 1);
            H = [];
        case 'unscented'
            [F, Q, R, W] = build_unscented_kf(F_los, Q_los, aug_data, general_config, sampling_interval);
            Hj_handle = [];
            H = [];
        case 'cubature'
            error('MATLAB:NotImplemented', 'Cubature KF variant is not implemented yet.');
        otherwise
            error('MATLAB:UndefinedKFType', 'KF type %s is not supported.', general_config.kf_type);
    end
    
    % 5. Store the results in the output struct under the scintillation model field.
    model_name = general_config.scintillation_training_data_config.scintillation_model;
    geometry_aided_measurement_config = struct();
    if isfield(general_config, 'geometry_aided_measurement_config')
        geometry_aided_measurement_config = general_config.geometry_aided_measurement_config;
    end
    kalman_pll_config.(model_name).(kf_type) = struct( ...
        'F', F, ...
        'Q', Q, ...
        'H', H, ...
        'R', R, ...
        'W', W, ...
        'Hj_handle', Hj_handle, ...
        'geometry_aided_measurement_config', geometry_aided_measurement_config, ...
        'augmentation_model_initializer', general_config.augmentation_model_initializer ...
    );
end
