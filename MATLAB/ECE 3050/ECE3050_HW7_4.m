%% Sampling and Reconstruction Problem
% Xc(jω) is a triangular spectrum from -3 to 3 rad/s
clear all; close all; clc;

%% Part (a): Plot Xc(jω)
fprintf('=== Part (a): Continuous-Time Fourier Transform ===\n\n');

% Define frequency axis for continuous-time spectrum
omega = linspace(-6, 6, 1000);

% Define Xc(jω) according to the given piecewise function
Xc = zeros(size(omega));
for i = 1:length(omega)
    w = omega(i);
    if w >= -3 && w < 0
        Xc(i) = 1 + w/3;
    elseif w >= 0 && w < 3
        Xc(i) = 1 - w/3;
    else
        Xc(i) = 0;
    end
end

% Plot Xc(jω)
figure('Position', [100, 100, 1400, 400]);
subplot(1,3,1);
plot(omega, Xc, 'b', 'LineWidth', 2);
grid on;
xlabel('\omega (rad/s)', 'FontSize', 12);
ylabel('X_c(j\omega)', 'FontSize', 12);
title('Part (a): Continuous-Time Spectrum X_c(j\omega)', 'FontSize', 14);
xlim([-6, 6]);
ylim([-0.2, 1.2]);

% Add annotations
hold on;
xline(0, 'k--', 'LineWidth', 0.5);
yline(0, 'k--', 'LineWidth', 0.5);

fprintf('Xc(jω) is a triangular spectrum:\n');
fprintf('  - Bandwidth: |ω| < 3 rad/s\n');
fprintf('  - Maximum frequency: ω_max = 3 rad/s\n');
fprintf('  - Peak magnitude: 1 at ω = 0\n\n');

%% Part (b): Plot Xs(jω) - Sampled Signal Spectrum
fprintf('=== Part (b): Sampled Signal Spectrum ===\n\n');

% Sampling parameters
Ts = pi/2;  % Sampling period
omega_s = 2*pi/Ts;  % Sampling frequency = 4 rad/s
fs_equiv = omega_s/(2*pi);  % Equivalent frequency in Hz

fprintf('Sampling period: Ts = π/2 ≈ %.4f s\n', Ts);
fprintf('Sampling frequency: ω_s = 2π/Ts = 4 rad/s\n');
fprintf('Nyquist frequency: ω_s/2 = 2 rad/s\n');
fprintf('Signal bandwidth: ω_max = 3 rad/s\n');
fprintf('ALIASING OCCURS: ω_max (3) > ω_s/2 (2) ✗\n\n');

% Xs(jω) consists of shifted and scaled replicas of Xc(jω)
% Xs(jω) = (1/Ts) * sum of Xc(j(ω - k*ω_s)) for all k

omega_plot = linspace(-10, 10, 2000);
Xs = zeros(size(omega_plot));

% Sum replicas (use k = -3 to 3 for sufficient coverage)
for k = -3:3
    for i = 1:length(omega_plot)
        w = omega_plot(i);
        w_shifted = w - k*omega_s;
        
        % Evaluate Xc at shifted frequency
        if w_shifted >= -3 && w_shifted < 0
            Xs(i) = Xs(i) + (1/Ts) * (1 + w_shifted/3);
        elseif w_shifted >= 0 && w_shifted < 3
            Xs(i) = Xs(i) + (1/Ts) * (1 - w_shifted/3);
        end
    end
end

% Plot Xs(jω)
subplot(1,3,2);
plot(omega_plot, Xs, 'r', 'LineWidth', 2);
grid on;
xlabel('\omega (rad/s)', 'FontSize', 12);
ylabel('X_s(j\omega)', 'FontSize', 12);
title('Part (b): Sampled Signal Spectrum X_s(j\omega)', 'FontSize', 14);
xlim([-10, 10]);
ylim([-0.1, 1.5]);

% Mark replica centers
hold on;
for k = -2:2
    xline(k*omega_s, 'k--', 'Alpha', 0.3);
    if k ~= 0
        text(k*omega_s, 1.4, sprintf('k=%d', k), 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end
xline(0, 'k--', 'LineWidth', 1);
yline(0, 'k--', 'LineWidth', 0.5);

% Highlight aliasing region
patch([-2, 2, 2, -2], [-0.1, -0.1, 1.5, 1.5], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text(0, 1.3, 'Base band', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'g');

fprintf('Xs(jω) shows overlapping replicas → ALIASING\n\n');

%% Part (c): Plot DTFT Xd(e^jω̂)
fprintf('=== Part (c): DTFT of Discrete Sequence ===\n\n');

% The DTFT is related to Xs(jω) by: Xd(e^jω̂) = Xs(jω)|_{ω=ω̂/Ts}
% where ω̂ = ωTs is the normalized (discrete) frequency
% DTFT is 2π-periodic

omega_hat = linspace(-2*pi, 2*pi, 2000);
Xd = zeros(size(omega_hat));

% Compute DTFT (one period is same as Xs scaled)
for i = 1:length(omega_hat)
    w_hat = omega_hat(i);
    w = w_hat / Ts;  % Convert discrete frequency to continuous frequency
    
    % Evaluate Xs(jω) at this frequency
    % Since Xs is periodic with period ω_s, we only need to evaluate base replicas
    for k = -3:3
        w_shifted = w - k*omega_s;
        
        if w_shifted >= -3 && w_shifted < 0
            Xd(i) = Xd(i) + (1/Ts) * (1 + w_shifted/3);
        elseif w_shifted >= 0 && w_shifted < 3
            Xd(i) = Xd(i) + (1/Ts) * (1 - w_shifted/3);
        end
    end
end

% Plot DTFT
subplot(1,3,3);
plot(omega_hat/pi, Xd, 'm', 'LineWidth', 2);
grid on;
xlabel('\omega-hat (×π rad)', 'FontSize', 12);
ylabel('X_d(e^{j\omega-hat})', 'FontSize', 12);
title('Part (c): DTFT X_d(e^{j\omega-hat})', 'FontSize', 14);
xlim([-2, 2]);
ylim([-0.1, 1.5]);

% Mark period boundaries
hold on;
xline(-1, 'k--', 'Alpha', 0.5, 'LineWidth', 1.5);
xline(1, 'k--', 'Alpha', 0.5, 'LineWidth', 1.5);
xline(0, 'k--', 'LineWidth', 1);
yline(0, 'k--', 'LineWidth', 0.5);

% Annotate periodicity
text(-1, 1.4, '-π', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
text(0, 1.4, '0', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
text(1, 1.4, 'π', 'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');

patch([-1, 1, 1, -1], [-0.1, -0.1, 1.5, 1.5], 'c', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text(0, 1.25, 'One period', 'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', 'c');

fprintf('DTFT is 2π-periodic\n');
fprintf('Normalized frequency: ω̂ = ωTs = ω(π/2)\n');
fprintf('ω̂ = π corresponds to ω = 2 rad/s\n\n');

%% Part (d): Reconstruction Analysis
fprintf('=== Part (d): Reconstruction Analysis ===\n\n');

% Detailed Console Output
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('RECONSTRUCTION ANALYSIS - DETAILED ANSWER\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

fprintf('Can we reconstruct x_c(t) from x_s(t) or x_d[n]?\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('\n❌ NO - Perfect reconstruction is IMPOSSIBLE\n\n');

fprintf('Justification:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('1. Signal bandwidth: ω_max = 3 rad/s\n');
fprintf('2. Sampling frequency: ω_s = 2π/Ts = 2π/(π/2) = 4 rad/s\n');
fprintf('3. Nyquist frequency: ω_s/2 = 2 rad/s\n');
fprintf('4. Nyquist criterion: Requires ω_s > 2ω_max\n');
fprintf('   → Need: ω_s > 6 rad/s\n');
fprintf('   → Have: ω_s = 4 rad/s ✗\n\n');
fprintf('5. ALIASING occurs because ω_max (3) > ω_s/2 (2)\n\n');

fprintf('Methods to Enable Reconstruction:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

fprintf('METHOD 1: Increase Sampling Rate\n');
fprintf('────────────────────────────────\n');
fprintf('• Required: Ts < π/ω_max = π/3 ≈ 1.047\n');
fprintf('• Current: Ts = π/2 ≈ 1.571 (TOO LARGE)\n');
fprintf('• Example: Use Ts = π/4, then ω_s = 8 rad/s > 6 rad/s ✓\n');
fprintf('• Then use ideal lowpass filter: H(jω) = Ts for |ω| < ω_s/2, else 0\n\n');

fprintf('METHOD 2: Pre-filter (Anti-aliasing)\n');
fprintf('────────────────────────────────────\n');
fprintf('• Apply ideal lowpass filter BEFORE sampling\n');
fprintf('• Cutoff frequency: ω_c = ω_s/2 = 2 rad/s\n');
fprintf('• This removes spectral content for |ω| > 2\n');
fprintf('• Can reconstruct the FILTERED signal (not original)\n');
fprintf('• Original information for 2 < |ω| < 3 is permanently lost\n\n');

fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');