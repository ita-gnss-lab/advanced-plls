% plot_rmse_results.m
%
% Description:
%   Script to plot RMSE results for CSM and CPSSM (without and with refraction)
%
% Author: Rodrigo de Lima Florindo
% ORCID: https://orcid.org/0000-0003-0412-5583
% Email: rdlfresearch@gmail.com

% Load data
data_file = fullfile('results', 'geometry_aided.mat');
if ~isfile(data_file)
    error('Data file not found: %s', data_file);
end
load(data_file, 'results', 'sigma2_W_3_sweep');

% Plot settings
font_size = 22;  % Base font size for labels, legends, and ticks

% Define model names and approaches
model_names = {'cpssm_wo_refr', 'cpssm_w_refr'};
approaches_all = {'kf_ar'}; % , 'akf_ar', 'ahl_kf_ar', 'kf', 'akf'
approach_labels_all = {'KF-AR'}; % , 'AKF-AR', 'AHL-KF-AR', 'KF', 'AKF'
approaches_ar = {'kf_ar'}; % , 'akf_ar', 'ahl_kf_ar'
approach_labels_ar = {'KF-AR'}; % , 'AKF-AR', 'AHL-KF-AR'
severities = {'weak', 'strong'};
field_names = {'phi_T', 'phi_W', 'phi_AR'};
colors = turbo(numel(approach_labels_all));
% Loop through each model
for i_model = 1:numel(model_names)
    model = model_names{i_model};
    % Create figure with larger width and height
    fig = figure('Name', sprintf('%s MRMSE', upper(model)), ...
                 'Units', 'normalized', ...
                 'Position', [0.05 0.05 1 0.8]);
                 % 'Color', 'w', ...

    % Set default font and LaTeX interpreter
    set(groot, ...
        'defaultAxesFontName', 'Times New Roman', ...
        'defaultTextFontName', 'Times New Roman', ...
        'defaultTextInterpreter', 'latex', ...
        'defaultAxesTickLabelInterpreter', 'latex');

    tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Plot rows (phi_T, phi_W, phi_AR) and columns (weak, strong)
    for row = 1:3
        for col = 1:2
            ax = nexttile;
            hold(ax, 'on');
            set(ax, 'FontName', 'Times New Roman', 'FontSize', font_size);
            severity = severities{col};
            field = field_names{row};

            % Determine which approaches and labels to use
            if row < 3
                approaches = approaches_all;
                approach_labels = approach_labels_all;
            else
                approaches = approaches_ar;
                approach_labels = approach_labels_ar;
            end

            % Plot each approach
            for k = 1:numel(approaches)
                approach = approaches{k};
                data_matrix = results.(model).(severity).(approach).(field);
                mean_rmse = mean(data_matrix, 2);
                plot(sigma2_W_3_sweep, mean_rmse, 'LineWidth', 2); % 'Color', colors(k,:)
            end

            % Title for first row
            if row == 1
                severity_label = [upper(severity(1)), severity(2:end)];
                title(severity_label, 'FontWeight', 'normal', 'FontSize', font_size);
            end

            % Y-label for first column
            if col == 1
                phi_label = ['\mathrm{',field(5:end),'}']; % 'T', 'W', or 'AR'
                ylabel(sprintf('MRMSE($\\hat{\\phi}_{%s}[k|k-1]$)', phi_label), 'Interpreter', 'latex', 'FontSize', font_size, 'FontName', 'Times New Roman');
            end

            % X-label for bottom row
            if row == 3
                xlabel('$\sigma^2_{\mathrm{W}}$', 'Interpreter', 'latex', 'FontSize', font_size, 'FontName', 'Times New Roman');
            end

            % Legend only on the first tile
            if row == 2 && col == 1
                legend(approach_labels, 'Location', 'best', 'FontSize', font_size - 2, 'FontName', 'Times New Roman');
            end
            
            xlim([1e-14,1e2])
            set(ax, 'XScale', 'log');
            grid(ax, 'on');
            grid("minor");
        end
    end

    % Save figures
    out_base = fullfile('results', sprintf('%s_mrmse', model));
    exportgraphics(fig, [out_base, '.pdf'], 'ContentType', 'vector');
    savefig(fig, [out_base, '.fig']);
end
