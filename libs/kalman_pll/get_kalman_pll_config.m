function [kalman_pll_config, initial_estimates] = get_kalman_pll_config(general_config, cache_dir, is_enable_cmd_print)
% get_kalman_pll_config
%
% Syntax:
%   [kalman_pll_config, initial_estimates] = get_kalman_pll_config(general_config, cache_dir, is_enable_cmd_print)
%
% Description:
%   Computes or retrieves the Kalman filter settings and initial state estimates
%   based on the provided configuration. Caching is used to avoid recomputation.
%   If caching is disabled or the cache is invalid, new parameters are computed
%   and the cache is updated.
%
% Inputs:
%   general_config - Struct containing configuration settings:
%       kf_type: Char {'standard', 'extended', 'unscented', 'cubature'}
%       discrete_wiener_model_config: Cell array {L, M, sampling_interval, sigma, delta}
%           L        - Number of carriers (positive integer scalar)
%           M        - Order of the Wiener process (positive integer scalar)
%           sampling_interval - Sampling interval for LOS dynamics (positive scalar)
%           sigma    - Numeric vector of variances; length must equal (L + M - 1)
%           delta    - Numeric vector of normalized frequencies; length must equal L
%
%       scintillation_training_data_config: Struct with scintillation model settings.
%           scintillation_model - String indicating the model ('CSM', 'TPPSM', or 'NONE')
%           For CSM:
%               S4              - Scintillation index (numeric scalar in [0,1])
%               tau0            - Signal decorrelation time (positive scalar)
%               simulation_time - Duration of simulation (positive scalar)
%               sampling_interval - Sampling interval (positive scalar)
%           For TPPSM:
%               scenario        - String: 'weak', 'moderate', or 'strong'
%               simulation_time - Duration of simulation (positive scalar)
%               sampling_interval - Sampling interval (positive scalar)
%               is_refractive_effects_removed - Logical flag (true/false; defaults to false)
%
%       C_over_N0_array_dBHz - Numeric vector (positive) of baseline C/N0 values (in dBHz)
%                              % NOTE: For now, the code only comprises
%                              single frequency tracking. In this case, 
%                              this variable should only be 
%                              configured as a scalar instead of an array.
%
%       initial_states_distributions_boundaries - Non-empty cell array; each cell contains a 1x2 numeric vector
%           specifying lower and upper bounds (first element < second element).
%
%       expected_doppler_profile - Non-empty numeric vector of Doppler profile values.
%
%       augmentation_model_initializer - Struct specifying the augmentation model initialization method and its parameters:
%           Fields:
%               id - A string indicating the augmentation model initialization method. Allowed values are:
%                    'arfit', 'aryule', 'kinematic', 'arima', 'rbf', or 'none'.
%               model_params - A struct containing parameters specific to the chosen method:
%                  - For 'arfit' and 'aryule': include the field 'model_order' (numeric) indicating the order of the VAR model.
%                  - For 'kinematic': include 'wiener_mdl_order' (numeric) and 'process_noise_variance' (numeric) to define the LOS augmentation.
%                  - For 'arima': include the fields 'p', 'D', and 'q' corresponding to the ARIMA model parameters.
%                  - For 'rbf': include the necessary parameters (e.g., 'neurons_amount'); note that this method is under development.
%                  - For 'none': no additional parameters are required.
%
%       is_use_cached_settings - Boolean flag to use cached configurations if available.
%       is_generate_random_initial_estimates - Boolean flag to generate initial estimates randomly.
%
%   cache_dir - String specifying the directory where cache files are stored.
%
%   is_enable_cmd_print - Logical flag for enabling command-line prints.
%
% Outputs:
%   kalman_pll_config - Struct containing the Kalman state space matrices.
%
%   initial_estimates - Column vector of initial state estimates. The number of rows
%                       equals the number of states in the state transition matrix.
%
% Notes:
%   - The sampling interval in scintillation_training_data_config is compared against
%     the third element in discrete_wiener_model_config. They must match.
%   - Caching is performed using get_cached_kalman_pll_config, update_cache, and
%     get_initial_estimates.
%
% Example:
%   cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');
%   training_data_config = struct('scintillation_model', 'none', 'sampling_interval', sampling_interval);
%   general_config = struct( ...
%       'kf_type', 'standard', ...
%       'discrete_wiener_model_config', { {1, 3, 0.01, [0,0,1e-2], 1} }, ...
%       'scintillation_training_data_config', training_data_config, ...
%       'C_over_N0_array_dBHz', 35, ...
%       'initial_states_distributions_boundaries',{ {[-pi,pi], [-25,25], [-0.1,0.1]} }, ...
%       'expected_doppler_profile', [0, 1000, 0.94], ...
%       'augmentation_model_initializer', struct('id', 'aryule', 'model_params', struct('model_order', 3)), ...
%       'is_use_cached_settings', false, ...
%       'is_generate_random_initial_estimates', true ...
%       'is_enable_cmd_print', false ...
%   );
%   [kalman_pll_config, initial_estimates] = get_kalman_pll_config(general_config, 'cache', true);
%
% Dependencies:
%   get_cached_kalman_pll_config, update_cache, get_initial_estimates
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    % Validate configurations using helper functions.
    [~, ~, sampling_interval_dw, ~, ~] = validate_discrete_wiener_model_config(general_config.discrete_wiener_model_config);

    scint_config = validateScintillationTrainingDataConfig(general_config.scintillation_training_data_config);
    validateattributes(general_config.C_over_N0_array_dBHz, {'numeric'}, {'nonempty','vector','positive'}, mfilename, 'C_over_N0_array_dBHz');
    validate_initial_states_boundaries(general_config.initial_states_distributions_boundaries);
    validateattributes(general_config.expected_doppler_profile, {'numeric'}, {'nonempty','vector'}, mfilename, 'expected_doppler_profile');
    validate_augmentation_model(general_config);
    validateattributes(cache_dir, {'char', 'string'}, {'nonempty'}, mfilename, 'cache_dir');
    validateattributes(is_enable_cmd_print, {'logical'}, {'scalar'}, mfilename, 'is_enable_cmd_print');

    % New validation for the kf_type field.
    allowed_kf_types = {'standard', 'extended', 'unscented', 'cubature'};
    validateattributes(general_config.kf_type, {'char','string'}, {'nonempty'}, mfilename, 'kf_type');
    if ~ismember(lower(string(general_config.kf_type)), allowed_kf_types)
        error('get_kalman_pll_config:InvalidKFType', ...
            'kf_type must be one of %s, but received: %s', strjoin(allowed_kf_types, ', '), general_config.kf_type);
    end

    % Check consistency between sampling intervals.
    if abs(scint_config.sampling_interval - sampling_interval_dw) > 1e-10
        error('get_kalman_pll_config:SamplingIntervalMismatch', ...
            'Sampling intervals in scintillation_training_data_config and discrete_wiener_model_config must match.');
    end

    % Ensure the cache directory exists.
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    cache_file = fullfile(cache_dir, 'kalman_pll_cache.mat');
    
    % Retrieve/update cached Kalman PLL configuration.
    [kalman_pll_config, is_cache_used] = get_cached_kalman_pll_config(general_config, cache_file, is_enable_cmd_print);
    kalman_pll_config = update_cache(general_config, cache_file, kalman_pll_config, is_cache_used, is_enable_cmd_print);
    initial_estimates = get_initial_estimates(general_config, kalman_pll_config);
end

%% Helper Functions
function [L, M, sampling_interval, sigma, delta] = validate_discrete_wiener_model_config(dw_config)
    % Validate that dw_config is a cell array with 5 elements.
    validateattributes(dw_config, {'cell'}, {'numel', 5}, mfilename, 'general_config.discrete_wiener_model_config');
    L = dw_config{1};
    M = dw_config{2};
    sampling_interval = dw_config{3};
    sigma = dw_config{4};
    delta = dw_config{5};
    
    % Validate L and M.
    validateattributes(L, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'L');
    validateattributes(M, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'M');
    
    % Validate sampling_interval.
    validateattributes(sampling_interval, {'numeric'}, {'scalar', 'positive'}, mfilename, 'sampling_interval');
    
    % Validate sigma and delta as numeric vectors.
    validateattributes(sigma, {'numeric'}, {'vector'}, mfilename, 'sigma');
    validateattributes(delta, {'numeric'}, {'vector'}, mfilename, 'delta');
    
    % Check dimensions.
    if numel(sigma) ~= (L + M - 1)
        error('general_config.discrete_wiener_model_config: The ''sigma'' element must be a numeric vector of length L+M-1, but it has %d elements.', numel(sigma));
    end
    if numel(delta) ~= L
        error('general_config.discrete_wiener_model_config: The ''delta'' element must be a numeric vector of length L, but it has %d elements.', numel(delta));
    end
end

function scint_config = validateScintillationTrainingDataConfig(scint_config)
    % Validate that scint_config is a nonempty struct.
    validateattributes(scint_config, {'struct'}, {'nonempty'}, mfilename, 'general_config.scintillation_training_data_config');
    validateattributes(scint_config.scintillation_model, {'char','string'}, {'nonempty'}, mfilename, 'general_config.scintillation_training_data_config.scintillation_model');
    
    % For models other than 'NONE', common fields must be present.
    if ~strcmp(scint_config.scintillation_model, 'none')
        actual_fields = fieldnames(scint_config);
        common_fields = {'simulation_time', 'sampling_interval', 'is_unwrapping_used'};
        [found, ~] = ismember(common_fields, actual_fields);
        %TODO: Refactor the test unit to comprehend this new validation
        if ~all(found)
            missing_fields = common_fields(~found);
            error("MATLAB:missing_inputs_scintillation_training_data_config",...
                  "Inputs missing: %s", strjoin(missing_fields, ', '));
        end
        validateattributes(scint_config.simulation_time, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'nonnan'}, mfilename, 'simulation_time');
        validateattributes(scint_config.sampling_interval, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'nonnan'}, mfilename, 'sampling_interval');
        validateattributes(scint_config.is_unwrapping_used, {'logical'}, ...
            {'nonempty'}, mfilename, 'is_unwrapping_used');
    end
    
    % Model-specific validations.
    switch upper(scint_config.scintillation_model)
        case 'CSM'
            validateattributes(scint_config.S4, {'numeric'}, ...
                {'scalar', 'real', '>=', 0, '<=', 1}, mfilename, 'S4');
            validateattributes(scint_config.tau0, {'numeric'}, ...
                {'scalar', 'positive', 'finite', 'nonnan'}, mfilename, 'tau0');
            
        case 'TPPSM'
            if ~isfield(scint_config, 'scenario')
                error('validateScintConfig:MissingTPPSMField', ...
                    'For TPPSM, the field ''scenario'' is required in general_config.scintillation_training_data_config.');
            end
            validateattributes(scint_config.scenario, {'char','string'}, {'nonempty'}, mfilename, 'scenario');
            scenario_val = upper(strtrim(scint_config.scenario));
            if ~ismember(scenario_val, {'WEAK', 'MODERATE', 'STRONG'})
                error('validateScintConfig:InvalidScenario', ...
                    'Field ''scenario'' must be one of ''Weak'', ''Moderate'', or ''Severe''.');
            end
            if ~isfield(scint_config, 'is_refractive_effects_removed')
                scint_config.is_refractive_effects_removed = false;
            else
                validateattributes(scint_config.is_refractive_effects_removed, {'logical'}, {'scalar'}, mfilename, 'is_refractive_effects_removed');
            end
            
        case 'NONE'
            % No further validation required.
        otherwise
            error('validateScintConfig:UnsupportedModel', 'Unsupported scintillation model: %s', model);
    end
end

function validate_initial_states_boundaries(boundaries)
    % Validate that boundaries is a non-empty cell array and each cell
    % contains a 1x2 numeric vector with the first element less than the second.
    if ~iscell(boundaries) || isempty(boundaries)
        error('get_kalman_pll_config:InvalidBoundaries', 'initial_states_distributions_boundaries must be a non-empty cell array.');
    end
    for i = 1:length(boundaries)
        b = boundaries{i};
        validateattributes(b, {'numeric'}, {'vector', 'numel', 2}, mfilename, 'initial_states_distributions_boundaries');
        if b(1) >= b(2)
            error('get_kalman_pll_config:InvalidBoundaries', 'Each boundary must have its first element less than its second.');
        end
    end
end

function validate_augmentation_model(general_config)
    % Validate that the augmentation model initializer is a nonempty string and one of the allowed values.
    validateattributes(general_config.augmentation_model_initializer, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer');
    if ~any(strcmpi(general_config.augmentation_model_initializer.id, {'arfit', 'aryule', 'arima', 'rbf', 'kinematic', 'none'}))
        error('get_kalman_pll_config:InvalidAugmentationModel', ...
            'augmentation_model_initializer must be ''arfit'', ''aryule'', ''second_wiener_mdl'', ''rbf'' or ''none''. Received: `%s`.', model_initializer.id);
    end
    validateattributes(general_config.augmentation_model_initializer.model_params, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params');
    switch general_config.augmentation_model_initializer.id
        case 'arfit'
            validateattributes(general_config.augmentation_model_initializer.model_params, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params');
            validateattributes(general_config.augmentation_model_initializer.model_params.model_order, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.model_order');
        case 'aryule'
            if general_config.discrete_wiener_model_config{1} ~= 1
                error('get_kalman_pll_config:incompatible_model_with_multi_frequency_tracking', ...
                    ['The aryule model_initializer is not compatible with multi-frequency carrier phase tracking systems. ' ...
                    'This happens because the function `aryule` is restricted to the scalar case.']);
            end
            validateattributes(general_config.augmentation_model_initializer.model_params, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params');
            validateattributes(general_config.augmentation_model_initializer.model_params.model_order, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.model_order');
        case 'kinematic'
            %NOTE: For now, we're only considering the single-frequency
            %scenario. This will probably need to change later.
            validateattributes(general_config.augmentation_model_initializer.model_params, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params');
            validateattributes(general_config.augmentation_model_initializer.model_params.wiener_mdl_order, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.wiener_mdl_order');
            validateattributes(general_config.augmentation_model_initializer.model_params.process_noise_variance, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.process_noise_variance');
        case 'arima'
            validateattributes(general_config.augmentation_model_initializer.model_params, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params');
            validateattributes(general_config.augmentation_model_initializer.model_params.p, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.p');
            validateattributes(general_config.augmentation_model_initializer.model_params.D, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.D');
            validateattributes(general_config.augmentation_model_initializer.model_params.q, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.Q');
        case 'rbf'
            error("MATLAB:RBFUnavailable","RBF model initializer is still under development.")
            % These validations below should be used later when RBF module
            % is developed.
            %validateattributes(model_initializer.model_params, {'struct'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params');
            %validateattributes(model_initializer.model_params.neurons_amount, {'double'}, {'nonempty'}, mfilename, 'augmentation_model_initializer.model_params.neurons_amount');
    end
end
