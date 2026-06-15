%% Initial parameters
fortaleza_lla = [-3.73, -38.52, 21]; % [deg, deg, m] % see: my Thesis
stop_simulation_time = to_down_datetime+seconds(300); % end simulation time
freq = "L1";
constellation = "gps";

out = cpssm('rx_origin', fortaleza_lla, 'constellation', constellation, 'frequency', freq);

n_scenarios = numel(out.gps.scenario);
geometry_info = repmat(struct( ...
    'azimuth', [], ...
    'zenith', [], ...
    'rho_true', [], ... % geometric range [m]
    'time', [], ...
    'svid', []), ...
    1, n_scenarios);

out_dir = "geometry_csv";
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

for i=1:n_scenarios
    scenario=out.gps.scenario(i);

    % get azimuth angle, elevation angle, geometric distance, and time for
    % this sat-rx scenario
    [az, el, rho, t] = aer(scenario.rx, scenario.sat);
    
    geometry_info(i).azimuth = az;
    geometry_info(i).zenith = 90-el;
    geometry_info(i).rho_true = rho;
    geometry_info(i).time = t;
    geometry_info(i).svid = scenario.sat.Name;

    svid = string(geometry_info(i).svid);

    % Make safe filename
    filename = regexprep(svid, '[^\w\-]', '_') + ".csv";
    filepath = fullfile(out_dir, filename);

    % Force column vectors
    zenith = geometry_info(i).zenith(:);
    azimuth = geometry_info(i).azimuth(:);

    T = table(zenith, azimuth);
    writetable(T, filepath);
end

save(fullfile(out_dir, "geometry_info.mat"), "geometry_info");
