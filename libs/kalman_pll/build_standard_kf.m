function [F, Q, H, R, W] = build_standard_kf(F_los, Q_los, aug_data, general_config, sampling_interval)
% build_standard_kf 
% 
% constructs the full standard Kalman filter matrices.
%
% Inputs:
%   F_los, Q_los  - LOS dynamics matrices.
%   aug_data       - Struct returned by compute_augmentation.
%   general_config- Configuration struct containing, among others, C_over_N0_array_dBHz.
%   sampling_interval - The sampling interval.
%
% Outputs:
%   F, Q, H, R, W - The state transition, process noise, measurement, measurement noise,
%                   and additional matrices for the Kalman filter.
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

    % Default measurement noise covariance:
    phase_noise_variance = get_phase_variances(general_config.C_over_N0_array_dBHz, sampling_interval);
    is_geometry_aided = isfield(general_config, 'geometry_aided_measurement_config') && ...
        isfield(general_config.geometry_aided_measurement_config, 'is_used') && ...
        general_config.geometry_aided_measurement_config.is_used;
    if is_geometry_aided
        geometry_noise_variance = general_config.geometry_aided_measurement_config.noise_variance;
        R = diag([phase_noise_variance, geometry_noise_variance]);
    else
        R = diag(phase_noise_variance);
    end

    switch lower(string(aug_data.augmentation_type))
        case 'none'
            % No augmentation is applied.
            F = F_los;
            Q = Q_los;
            % Construct H for the LOS part.
            H = [1, zeros(1, size(F_los,1)-1)];
            W = zeros(size(F_los,1), 1);

        case {'arfit', 'aryule'}
            
            F_aug = aug_data.F_aug;
            Q_aug = aug_data.Q_aug;
            
            % Extract extra information (for ARYULE, intercept is set to 0).
            intercept_vector = aug_data.intercept;
            states_amount = aug_data.augInfo.states_amount;
            model_order = aug_data.augInfo.model_order;

            % Validate inputs
            validateattributes(F_los, {'numeric'}, {'2d', 'nonempty'}, mfilename, 'F_los');
            validateattributes(Q_los, {'numeric'}, {'2d', 'nonempty', 'square'}, mfilename, 'Q_los');
            if ~isnan(model_order)
                validateattributes(F_aug, {'numeric'}, {'2d', 'nonempty'}, mfilename, 'F_aug');
                validateattributes(Q_aug, {'numeric'}, {'2d', 'nonempty', 'square'}, mfilename, 'Q_aug');
                validateattributes(model_order, {'numeric'}, {'scalar', 'integer', '>=', 1}, mfilename, 'model_order');
                validateattributes(states_amount, {'numeric'}, {'scalar', 'integer', '>=', 1}, mfilename, 'states_amount');
                % Ensure Q_var is symmetric and positive semi-definite
                if ~isequal(Q_aug, Q_aug') 
                    error('construct_kalman_matrices:QaugNotSymmetric', 'Q_aug must be symmetric.');
                end
                if any(eig(Q_aug) < -1e-10) 
                    error('construct_kalman_matrices:QaugNotPositiveSemiDefinite', 'Q_aug must be positive semi-definite.');
                end
            end
            
            % Ensure Q_los is symmetric and positive semi-definite
            if ~isequal(Q_los, Q_los') 
                error('construct_kalman_matrices:QlosNotSymmetric', 'Q_los must be symmetric.');
            end
            if any(eig(Q_los) < -1e-10) % Allow small numerical errors
                error('construct_kalman_matrices:QlosNotPositiveSemiDefinite', 'Q_los must be positive semi-definite.');
            end
            
            % Combine LOS and VAR state transitions
            F = blkdiag(F_los, F_aug);
            
            % Combine LOS and VAR covariance matrices
            Q = blkdiag(Q_los, Q_aug);
            
            % Construct measurement matrix H.
            % Legacy mode: one scalar phase measurement that observes LOS + AR phase.
            % Geometry-aided mode: a second row observes LOS phase only.
            H_total = [1, zeros(1, size(F_los,1)-1), 1, zeros(1, states_amount*model_order - 1)];
            H_los = [1, zeros(1, size(F_los,1)-1), zeros(1, states_amount*model_order)];
            if is_geometry_aided
                H = [H_total; H_los];
            else
                H = H_total;
            end
            
            % Construct the augmented intercept vector W.
            W = [zeros(size(F_los,1), 1); intercept_vector; zeros(states_amount*(model_order-1),1)];

        case 'kinematic'
            % For the kinematic augmentation, simply concatenate the LOS and Wiener augmentation matrices.
            F = blkdiag(F_los, aug_data.F_aug);
            Q = blkdiag(Q_los, aug_data.Q_aug);
            % Construct H by selecting the first element of the LOS part and the first element of the augmentation part.
            n_los = size(F_los, 1);
            n_aug = size(aug_data.F_aug, 1);
            H = [1, zeros(1, n_los-1), 1, zeros(1, n_aug-1)];
            W = zeros(n_los + n_aug, 1);

        case 'arima'
            % For the ARIMA model, combine the LOS and ARIMA matrices.
            F = blkdiag(F_los, aug_data.F_aug);
            Q = blkdiag(Q_los, aug_data.Q_aug);
            % Use the H vector computed in the ARIMA augmentation branch.
            H = [1, zeros(1, size(F_los,1)-1), aug_data.H_aug];
            W = zeros(size(F,1), 1);

        case 'arfit-complex-field'
            error('MATLAB:NotImplementedAugmentationModel', ...
                'The arfit-complex-field augmentation model is only supported by the unscented KF.');

        otherwise
            error('MATLAB:InvalidAugmentationModel', 'KF augmentation type %s is not supported in standard KF builder.', aug_data.augmentation_type);
    end
end
