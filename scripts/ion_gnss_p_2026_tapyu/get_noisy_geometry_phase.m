function geometry_los_phase = get_noisy_geometry_phase(true_los_phase, geometry_noise_variance)
    geometry_los_phase = true_los_phase + sqrt(geometry_noise_variance) * randn(size(true_los_phase));
end