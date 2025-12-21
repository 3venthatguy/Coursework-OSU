%% Aliasing Problem - Plotting Fourier Transforms with Phasors
% Three sinusoids sampled at Ts = 0.01
clear all; close all; clc;

%% Signal Parameters
% xc(t) = cos(40πt + π/4)   → f0 = 20 Hz
% yc(t) = cos(240πt + π/4)  → f0 = 120 Hz
% zc(t) = cos(160πt - π/4)  → f0 = 80 Hz

Ts = 0.01;                    % Sampling period
fs = 1/Ts;                    % Sampling frequency = 100 Hz
omega_s = 2*pi*fs;            % Sampling frequency in rad/s = 200π

% Signal frequencies (rad/s)
omega_x = 40*pi;              % 20 Hz
omega_y = 240*pi;             % 120 Hz
omega_z = 160*pi;             % 80 Hz

% Phases
phi_x = pi/4;
phi_y = pi/4;
phi_z = -pi/4;

fprintf('Signal Parameters:\n');
fprintf('xc(t): f = 20 Hz,  ω = 40π rad/s,  φ = π/4\n');
fprintf('yc(t): f = 120 Hz, ω = 240π rad/s, φ = π/4\n');
fprintf('zc(t): f = 80 Hz,  ω = 160π rad/s, φ = -π/4\n');
fprintf('Sampling: Ts = %.2f s, fs = %.0f Hz, ωs = %.0fπ rad/s\n\n', Ts, fs, omega_s/pi);

%% Part (a): Continuous-Time Fourier Transform
fprintf('=== Part (a): Continuous-Time Fourier Transforms ===\n\n');

figure('Position', [50, 50, 1400, 900]);

% Frequency axis for continuous-time
omega_ct = linspace(-300*pi, 300*pi, 10000);

% Create subplots for magnitude and phase
subplot(2,1,1);
hold on; grid on;

% Impulse locations and magnitudes for all three signals
impulse_mag = pi;  % Magnitude of each impulse

% Plot xc(t) - Blue
stem([-omega_x, omega_x], [impulse_mag, impulse_mag], 'b', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_x, impulse_mag + 0.3, sprintf('π·e^{-jπ/4}'), 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_x, impulse_mag + 0.3, sprintf('π·e^{jπ/4}'), 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Plot yc(t) - Red
stem([-omega_y, omega_y], [impulse_mag, impulse_mag], 'r', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_y, impulse_mag + 0.3, sprintf('π·e^{-jπ/4}'), 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_y, impulse_mag + 0.3, sprintf('π·e^{jπ/4}'), 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Plot zc(t) - Magenta
stem([-omega_z, omega_z], [impulse_mag, impulse_mag], 'm', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_z, impulse_mag + 0.3, sprintf('π·e^{jπ/4}'), 'Color', 'm', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_z, impulse_mag + 0.3, sprintf('π·e^{-jπ/4}'), 'Color', 'm', 'FontSize', 10, 'HorizontalAlignment', 'center');

xlabel('\omega (rad/s)', 'FontSize', 12);
ylabel('|X_c(j\omega)|, |Y_c(j\omega)|, |Z_c(j\omega)|', 'FontSize', 12);
title('Part (a): Magnitude of Continuous-Time Fourier Transforms', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-300*pi, 300*pi]);
ylim([0, 5]);
xticks([-omega_y, -omega_z, -omega_x, 0, omega_x, omega_z, omega_y]);
xticklabels({'-240π', '-160π', '-40π', '0', '40π', '160π', '240π'});
legend('X_c(j\omega) [20 Hz]', '', 'Y_c(j\omega) [120 Hz]', '', 'Z_c(j\omega) [80 Hz]', '', 'Location', 'northeast');

% Phase plot
subplot(2,1,2);
hold on; grid on;

% Plot phases for xc(t)
stem([-omega_x, omega_x], [-phi_x, phi_x], 'b', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_x, -phi_x - 0.3, sprintf('-π/4'), 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_x, phi_x + 0.3, sprintf('π/4'), 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Plot phases for yc(t)
stem([-omega_y, omega_y], [-phi_y, phi_y], 'r', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_y, -phi_y - 0.3, sprintf('-π/4'), 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_y, phi_y + 0.3, sprintf('π/4'), 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Plot phases for zc(t)
stem([-omega_z, omega_z], [phi_z, -phi_z], 'm', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_z, phi_z + 0.3, sprintf('π/4'), 'Color', 'm', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_z, -phi_z - 0.3, sprintf('-π/4'), 'Color', 'm', 'FontSize', 10, 'HorizontalAlignment', 'center');

xlabel('\omega (rad/s)', 'FontSize', 12);
ylabel('Phase (radians)', 'FontSize', 12);
title('Part (a): Phase of Continuous-Time Fourier Transforms', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-300*pi, 300*pi]);
ylim([-pi, pi]);
xticks([-omega_y, -omega_z, -omega_x, 0, omega_x, omega_z, omega_y]);
xticklabels({'-240π', '-160π', '-40π', '0', '40π', '160π', '240π'});
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});
yline(0, 'k--', 'LineWidth', 0.5);

%% Part (b): Sampled Signal Fourier Transform
fprintf('=== Part (b): Sampled Signal Fourier Transforms ===\n\n');

figure('Position', [100, 50, 1400, 900]);

% For sampled signals, we get replicas at ω ± k*ωs scaled by 1/Ts
scale_factor = 1/Ts;  % = 100

% Show replicas from k = -2 to k = 2
k_range = -2:2;

subplot(2,1,1);
hold on; grid on;

% Plot Xs(jω) - Blue
for k = k_range
    omega_shift = k * omega_s;
    stem([-(omega_x + omega_shift), (omega_x + omega_shift)], ...
         [scale_factor*pi, scale_factor*pi], 'b', 'LineWidth', 1.5, 'MarkerSize', 6);
end
% Label main replicas
text(-omega_x, scale_factor*pi + 30, '100π·e^{-jπ/4}', 'Color', 'b', 'FontSize', 9, 'HorizontalAlignment', 'center');
text(omega_x, scale_factor*pi + 30, '100π·e^{jπ/4}', 'Color', 'b', 'FontSize', 9, 'HorizontalAlignment', 'center');

% Plot Ys(jω) - Red (aliases to ±40π)
for k = k_range
    omega_shift = k * omega_s;
    stem([-(omega_y + omega_shift), (omega_y + omega_shift)], ...
         [scale_factor*pi, scale_factor*pi], 'r', 'LineWidth', 1.5, 'MarkerSize', 6);
end
% The k=-1 replica of 240π lands at 240π - 200π = 40π (ALIASING!)
text(-omega_x, scale_factor*pi + 60, '100π·e^{jπ/4} (aliased from 240π)', 'Color', 'r', 'FontSize', 8, 'HorizontalAlignment', 'center');
text(omega_x, scale_factor*pi + 60, '100π·e^{-jπ/4} (aliased from 240π)', 'Color', 'r', 'FontSize', 8, 'HorizontalAlignment', 'center');

% Plot Zs(jω) - Magenta (aliases to ±40π)
for k = k_range
    omega_shift = k * omega_s;
    stem([-(omega_z + omega_shift), (omega_z + omega_shift)], ...
         [scale_factor*pi, scale_factor*pi], 'm', 'LineWidth', 1.5, 'MarkerSize', 6);
end
% The k=1 replica of -160π lands at -160π + 200π = 40π (ALIASING!)
text(-omega_x, scale_factor*pi + 90, '100π·e^{-jπ/4} (aliased from 160π)', 'Color', 'm', 'FontSize', 8, 'HorizontalAlignment', 'center');
text(omega_x, scale_factor*pi + 90, '100π·e^{jπ/4} (aliased from 160π)', 'Color', 'm', 'FontSize', 8, 'HorizontalAlignment', 'center');

% Mark sampling frequency replicas
for k = k_range
    xline(k*omega_s, 'k--', 'Alpha', 0.3, 'LineWidth', 1);
    if k ~= 0
        text(k*omega_s, 380, sprintf('k=%d', k), 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

% Highlight base band
patch([-omega_s/2, omega_s/2, omega_s/2, -omega_s/2], [0, 0, 400, 400], ...
      'g', 'FaceAlpha', 0.05, 'EdgeColor', 'g', 'LineWidth', 2);
text(0, 370, 'Base band (-ωs/2 to ωs/2)', 'HorizontalAlignment', 'center', ...
     'FontSize', 10, 'Color', 'g', 'FontWeight', 'bold');

xlabel('\omega (rad/s)', 'FontSize', 12);
ylabel('|X_s(j\omega)|, |Y_s(j\omega)|, |Z_s(j\omega)|', 'FontSize', 12);
title('Part (b): Magnitude of Sampled Signal Fourier Transforms', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-500*pi, 500*pi]);
ylim([0, 400]);
xticks([-400*pi, -omega_s, -omega_s/2, 0, omega_s/2, omega_s, 400*pi]);
xticklabels({'-400π', '-200π (-ωs)', '-100π', '0', '100π', '200π (ωs)', '400π'});

% Phase plot for sampled signals
subplot(2,1,2);
hold on; grid on;

% Plot phases for Xs(jω)
for k = k_range
    omega_shift = k * omega_s;
    stem([-(omega_x + omega_shift), (omega_x + omega_shift)], ...
         [-phi_x, phi_x], 'b', 'LineWidth', 1.5, 'MarkerSize', 6);
end

% Plot phases for Ys(jω) - note phase flips due to aliasing
for k = k_range
    omega_shift = k * omega_s;
    % At baseband: k=-1 gives 240π - 200π = 40π with phase swap
    if k == -1
        stem([-(omega_y + omega_shift), (omega_y + omega_shift)], ...
             [phi_y, -phi_y], 'r', 'LineWidth', 1.5, 'MarkerSize', 6);
    else
        stem([-(omega_y + omega_shift), (omega_y + omega_shift)], ...
             [-phi_y, phi_y], 'r', 'LineWidth', 1.5, 'MarkerSize', 6);
    end
end

% Plot phases for Zs(jω)
for k = k_range
    omega_shift = k * omega_s;
    if k == 1
        stem([-(omega_z + omega_shift), (omega_z + omega_shift)], ...
             [-phi_z, phi_z], 'm', 'LineWidth', 1.5, 'MarkerSize', 6);
    else
        stem([-(omega_z + omega_shift), (omega_z + omega_shift)], ...
             [phi_z, -phi_z], 'm', 'LineWidth', 1.5, 'MarkerSize', 6);
    end
end

% Mark sampling frequency replicas
for k = k_range
    xline(k*omega_s, 'k--', 'Alpha', 0.3, 'LineWidth', 1);
end

% Highlight base band
patch([-omega_s/2, omega_s/2, omega_s/2, -omega_s/2], [-pi, -pi, pi, pi], ...
      'g', 'FaceAlpha', 0.05, 'EdgeColor', 'g', 'LineWidth', 2);

xlabel('\omega (rad/s)', 'FontSize', 12);
ylabel('Phase (radians)', 'FontSize', 12);
title('Part (b): Phase of Sampled Signal Fourier Transforms', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-500*pi, 500*pi]);
ylim([-pi-0.5, pi+0.5]);
xticks([-400*pi, -omega_s, -omega_s/2, 0, omega_s/2, omega_s, 400*pi]);
xticklabels({'-400π', '-200π (-ωs)', '-100π', '0', '100π', '200π (ωs)', '400π'});
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});
yline(0, 'k--', 'LineWidth', 0.5);

%% Part (c): DTFT
fprintf('=== Part (c): DTFT of Discrete Sequences ===\n\n');

figure('Position', [150, 50, 1400, 900]);

% For DTFT: ω̂ = ωTs, and DTFT is 2π-periodic
% The impulses at ω = ±40π in Xs(jω) appear at ω̂ = ±40π·Ts = ±0.4π

omega_hat_x = omega_x * Ts;  % = 0.4π
omega_hat_y = omega_y * Ts;  % = 2.4π → aliases to 0.4π in [-π, π]
omega_hat_z = omega_z * Ts;  % = 1.6π → aliases to -0.4π in [-π, π]

subplot(2,1,1);
hold on; grid on;

% Show one period: [-π, π]
% All three signals alias to ω̂ = ±0.4π

% Plot Xd(e^jω̂) - Blue
stem([-omega_hat_x, omega_hat_x]/pi, [scale_factor*pi, scale_factor*pi], ...
     'b', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_hat_x/pi, scale_factor*pi + 30, '100π·e^{-jπ/4}', 'Color', 'b', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_hat_x/pi, scale_factor*pi + 30, '100π·e^{jπ/4}', 'Color', 'b', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');

% Plot Yd(e^jω̂) - Red (aliased to ±0.4π)
stem([-omega_hat_x, omega_hat_x]/pi, [scale_factor*pi, scale_factor*pi], ...
     'r', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_hat_x/pi, scale_factor*pi + 60, '100π·e^{jπ/4}', 'Color', 'r', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_hat_x/pi, scale_factor*pi + 60, '100π·e^{-jπ/4}', 'Color', 'r', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');

% Plot Zd(e^jω̂) - Magenta (aliased to ±0.4π)
stem([-omega_hat_x, omega_hat_x]/pi, [scale_factor*pi, scale_factor*pi], ...
     'm', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_hat_x/pi, scale_factor*pi + 90, '100π·e^{-jπ/4}', 'Color', 'm', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_hat_x/pi, scale_factor*pi + 90, '100π·e^{jπ/4}', 'Color', 'm', ...
     'FontSize', 10, 'HorizontalAlignment', 'center');

% Mark period boundaries
xline(-1, 'k--', 'LineWidth', 2, 'Label', '-π', 'LabelVerticalAlignment', 'bottom');
xline(1, 'k--', 'LineWidth', 2, 'Label', 'π', 'LabelVerticalAlignment', 'bottom');
xline(0, 'k-', 'LineWidth', 1);

% Highlight one period
patch([-1, 1, 1, -1], [0, 0, 400, 400], 'c', 'FaceAlpha', 0.05, ...
      'EdgeColor', 'c', 'LineWidth', 2);
text(0, 370, 'One Period (2π-periodic)', 'HorizontalAlignment', 'center', ...
     'FontSize', 11, 'Color', 'c', 'FontWeight', 'bold');

xlabel('\omega-hat (×π rad)', 'FontSize', 12);
ylabel('|X_d(e^{j\omega-hat})|, |Y_d(e^{j\omega-hat})|, |Z_d(e^{j\omega-hat})|', 'FontSize', 12);
title('Part (c): Magnitude of DTFT', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-2, 2]);
ylim([0, 400]);
xticks([-2, -1, -0.4, 0, 0.4, 1, 2]);
xticklabels({'-2π', '-π', '-0.4π', '0', '0.4π', 'π', '2π'});
legend('X_d (20 Hz)', '', 'Y_d (120→20 Hz aliased)', '', 'Z_d (80→20 Hz aliased)', '', ...
       'Location', 'northeast');

% Phase plot for DTFT
subplot(2,1,2);
hold on; grid on;

% Plot phases - all at ω̂ = ±0.4π but with different phase values

% Xd phase
stem([-omega_hat_x, omega_hat_x]/pi, [-phi_x, phi_x], 'b', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_hat_x/pi, -phi_x - 0.3, '-π/4', 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_hat_x/pi, phi_x + 0.3, 'π/4', 'Color', 'b', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Yd phase (swapped due to aliasing from 240π)
stem([-omega_hat_x, omega_hat_x]/pi, [phi_y, -phi_y], 'r', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_hat_x/pi, phi_y + 0.35, 'π/4', 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_hat_x/pi, -phi_y - 0.35, '-π/4', 'Color', 'r', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Zd phase (swapped due to aliasing from 160π)
stem([-omega_hat_x, omega_hat_x]/pi, [-phi_z, phi_z], 'm', 'LineWidth', 2, 'MarkerSize', 8);
text(-omega_hat_x/pi - 0.15, -phi_z - 0.4, '-π/4', 'Color', 'm', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(omega_hat_x/pi + 0.15, phi_z + 0.4, 'π/4', 'Color', 'm', 'FontSize', 10, 'HorizontalAlignment', 'center');

% Mark period boundaries
xline(-1, 'k--', 'LineWidth', 2);
xline(1, 'k--', 'LineWidth', 2);
xline(0, 'k-', 'LineWidth', 1);
yline(0, 'k--', 'LineWidth', 0.5);

% Highlight one period
patch([-1, 1, 1, -1], [-pi-0.5, -pi-0.5, pi+0.5, pi+0.5], 'c', 'FaceAlpha', 0.05, ...
      'EdgeColor', 'c', 'LineWidth', 2);

xlabel('\omega-hat (×π rad)', 'FontSize', 12);
ylabel('Phase (radians)', 'FontSize', 12);
title('Part (c): Phase of DTFT', 'FontSize', 14, 'FontWeight', 'bold');
xlim([-2, 2]);
ylim([-pi-0.5, pi+0.5]);
xticks([-2, -1, -0.4, 0, 0.4, 1, 2]);
xticklabels({'-2π', '-π', '-0.4π', '0', '0.4π', 'π', '2π'});
yticks([-pi, -pi/2, 0, pi/2, pi]);
yticklabels({'-π', '-π/2', '0', 'π/2', 'π'});

%% Summary text
fprintf('\nALIASING SUMMARY:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('• xc(t) at 20 Hz: BELOW Nyquist (50 Hz) → NO aliasing\n');
fprintf('• yc(t) at 120 Hz: ABOVE Nyquist → Aliases to 20 Hz\n');
fprintf('• zc(t) at 80 Hz: ABOVE Nyquist → Aliases to 20 Hz\n');
fprintf('\nAll three discrete sequences appear to have the same\n');
fprintf('frequency (20 Hz), differing only in phase!\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');