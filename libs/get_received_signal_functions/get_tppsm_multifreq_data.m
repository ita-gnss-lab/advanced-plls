function [psi_cpssm, ps_realization, general_params, irr_params_set, seed] = get_tppsm_multifreq_data(scenario, varargin)
% get_tppsm_multifreq_data
%
% Generates multi-frequency (L1, L2 and L5) realizations of ionospheric 
% scintillation complex field time series using preset irregularity 
% parameters that characterizes the compact phase-screen-based 
% scintillation model (CPSSM) for a specified scintillation scenario.
%
% Syntax:
%   [psi_cpssm, ps_realization, general_params, irr_params_set, seed] = ...
%       get_tppsm_multifreq_data(scenario, 'simulation_time', simulation_time, ...
%                      'sampling_interval', sampling_interval, ...
%                      'general_params', general_params, ...
%                      'irr_params_set', irr_params_set, ...
%                      'seed', seed, ...
%                      'rhof_veff_ratio', rhof_veff_ratio)
%
% Inputs:
%   scenario         - A string specifying the scintillation scenario 
%                      ('weak', 'moderate', or 'strong').
%
% Optional Name-Value Pair Inputs:
%   'simulation_time'     - Duration of the simulation in seconds (default: 300 s).
%   'sampling_interval'   - Sampling time in seconds (default: 0.01 s).
%   'general_params'      - Structure containing simulation parameters. If empty,
%                           the function will call get_general_parameters() internally.
%   'irr_params_set'      - Structure containing preset irregularity parameters.
%                           If empty, the function will call get_irregularity_parameters().
%   'seed'                - Seed for random number generation (default: 1).
%   'rhof_veff_ratio'     - (Optional) A numeric value for the computed 
%                           rhof/veff ratio for L1. If provided (and nonempty), 
%                           the function will skip its internal computation.
%   'is_enable_cmd_print' - logical value that configures whether command
%                           lines would appear on the regarding of the 
%                           usage of external rhof_veff_ratio.
%
% Outputs:
%   psi_cpssm        - Scintillation field time series (complex), as a row vector.
%   ps_realization   - Detrended phase screen realization (row vector).
%   general_params   - Simulation parameters used for this run.
%   irr_params_set   - Preset irregularity parameters used.
%   seed             - Seed value used.
%
% Example:
%   [psi_tpwpsm, ps_realization] = get_tppsm_multifreq_data('moderate', 'seed', 42, 'rhof_veff_ratio', 0.35);
%
% Notes:
%   - This implementation is for single-frequency carrier phase tracking.
%   - The function does not perform any extrapolation of irregularity parameters;
%     instead, it uses preset parameters directly.
%   - If 'rhof_veff_ratio' is provided, it is used directly; otherwise, the
%     function computes it by calling get_rhof_veff_ratio(general_params).
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

% Input validation using inputParser
p = inputParser;
addRequired(p, 'scenario', @(x) ischar(x) || isstring(x));
addParameter(p, 'simulation_time', 300, @(x) validateattributes(x, {'numeric'}, {'scalar', 'real', 'positive'}));
addParameter(p, 'sampling_interval', 0.01, @(x) validateattributes(x, {'numeric'}, {'scalar', 'real', 'positive'}));
addParameter(p, 'general_params', [], @(x) isempty(x) || isstruct(x));
addParameter(p, 'irr_params_set', [], @(x) isempty(x) || isstruct(x));
addParameter(p, 'seed', 1, @(x) validateattributes(x, {'numeric'}, {'scalar', 'real'}));
addParameter(p, 'rhof_veff_ratio', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'is_enable_cmd_print', true, @(x) validateattributes(x, {'logical'}, {'nonempty'}));
parse(p, scenario, varargin{:});

% Validate the inputed scenario for simulation
scenario = validatestring(p.Results.scenario, {'weak', 'moderate', 'strong'}, mfilename, 'scenario');

% Instantiate all parameters for internal using
simulation_time = p.Results.simulation_time;
sampling_interval = p.Results.sampling_interval;
general_params = p.Results.general_params;
irr_params_set = p.Results.irr_params_set;
seed = p.Results.seed;
rhof_veff_ratio = p.Results.rhof_veff_ratio;
is_enable_cmd_print = p.Results.is_enable_cmd_print;

% Validate the `simulation_time` variable
if simulation_time < sampling_interval
    error('get_tppsm_data:simulationTimeSmallerThanSamplingInterval', ...
          'The simulation_time (%g s) is smaller than the sampling_interval (%g s).', simulation_time, sampling_interval);
end

% Handle the number of the simulation samples
num_samples_exact = simulation_time / sampling_interval;
num_samples_rounded = round(num_samples_exact);
if abs(num_samples_exact - num_samples_rounded) > eps
    warning('get_tppsm_data:NonIntegerRatioSamples', ...
            'simulation_time / sampling_interval is not an integer. Rounded from %.5g to %d samples.', num_samples_exact, num_samples_rounded);
end

% Get the general parameter if empty
if isempty(general_params)
    general_params = get_general_parameters();
end
general_params.simulation_time = simulation_time;
general_params.dt = sampling_interval;

% Get the irregularity parameters if empty
if isempty(irr_params_set)
    irr_params_set = get_irregularity_parameters();
end

% Select the irregularity parameters for the chosen scenario
irr_params = irr_params_set.(scenario);

% If the external rhof_veff_ratio is not provided, compute it.
if isempty(rhof_veff_ratio)
    % If the rhof_veff_ratio is empty, obtain it using the general_params
    % configurations.
    rhof_veff_ratio = get_rhof_veff_ratio(general_params);
else
    if is_enable_cmd_print
        % Display a message if an externally provided rhof_veff ratio is
        % being used.
        fprintf('Using externally provided rhof_veff_ratio: %g\n', rhof_veff_ratio);
    end
end

% Set a random seed
rng(seed);

% Extrapolate the irregularity parameters for L2 and L5.
[extrapolated_irr_params, rhof_veff_ratio_vector] = freq_extrapolate(irr_params,general_params,rhof_veff_ratio);

% Chars cell of frequency bands for accessing the extrapolated irregularity
% parameters from `extrapolated_irr_params`
frequency_keys = {'L1', 'L2', 'L5'};
psi_cpssm = zeros(num_samples_rounded, length(rhof_veff_ratio_vector));
ps_realization = zeros(num_samples_rounded, length(rhof_veff_ratio_vector));
for i = 1:length(rhof_veff_ratio_vector)
    [scint_field, ~, detrended_phase] = get_scintillation_time_series(general_params, extrapolated_irr_params.(frequency_keys{i}), rhof_veff_ratio_vector(i), seed);
    psi_cpssm(:,i) = scint_field(1:num_samples_rounded);
    ps_realization(:,i) = detrended_phase(1:num_samples_rounded);
end

end
