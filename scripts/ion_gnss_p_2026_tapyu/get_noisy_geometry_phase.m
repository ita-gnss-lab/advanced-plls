function geometry_los_phase = get_noisy_geometry_phase(true_los_phase, phi_LOS_noise_variance)
    geometry_los_phase = true_los_phase + sqrt(phi_LOS_noise_variance) * randn(size(true_los_phase));
end