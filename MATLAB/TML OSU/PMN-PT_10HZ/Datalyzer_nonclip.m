clear; clc; close all;

%% Load and Prepare Data
data = readmatrix("C:\Users\evanm\OneDrive\Desktop\260.csv");

time_sorted = data(:, 1);
CH1 = data(:, 2) * 10;  % Account for 10x amplifier
CH2 = data(:, 3);

% Apply moving average smoothing
window_size = 100; % Adjust this - larger = smoother but less detail
CH1_smooth = movmean(CH1, window_size);
CH2_smooth = movmean(CH2, window_size);

%% Voltage Input Hysteresis Loop

figure(1);
plot(CH1_smooth, CH2_smooth, 'b-', 'LineWidth', 1.5);
xlabel('CH1 Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CH2 Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
title('Hysteresis Loop', 'FontSize', 14, 'FontWeight', 'bold');
xline(0, 'k--', 'LineWidth', 0.8);
yline(0, 'k--', 'LineWidth', 0.8);


%% Non-Annotated Hysteresis Loop E-field vs Polarization

% Sample parameters (adjust to your actual values)
A = (1.58e-3) * (3.89e-3);          % Sample area (m²)
d = 0.5e-3;                           % Sample thickness (m)
C_ref = 1e-6;                       % Reference capacitor (F)

% Calculate E-field and Polarization
E_field = CH1_smooth / d;           % V/m
Q = C_ref * CH2_smooth;             % Coulombs
P = Q / A;                          % C/m²


figure(2);
plot(E_field, P, 'r-', 'LineWidth', 1);   % E in V/m, P in C/m²
hold on;

% Add time-spaced markers
time_interval = 0.003; % Time spacing in seconds (adjust this)
% Find indices corresponding to this time interval
time_diff = diff(time_sorted);
cumulative_time = [0; cumsum(time_diff)];

% Select points at regular time intervals
marker_times = 0:time_interval:max(cumulative_time);
marker_indices = zeros(length(marker_times), 1);

for i = 1:length(marker_times)
    [~, marker_indices(i)] = min(abs(cumulative_time - marker_times(i)));
end

% Remove duplicate indices
marker_indices = unique(marker_indices);

% Plot markers — E in V/m, P in C/m²
plot(E_field(marker_indices), P(marker_indices), ...
     'ro', 'MarkerSize', 9, 'MarkerFaceColor', 'r');

hold off;

xlabel('Electric Field (V m^{-1})', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Polarization (C m^{-2})', 'FontSize', 12, 'FontWeight', 'bold');
title('Ferroelectric Hysteresis Loop', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xline(0, 'k--', 'LineWidth', 0.8);
yline(0, 'k--', 'LineWidth', 0.8);
box on;

%% Annotated Hysteresis Loop E-field vs Polarization

figure(3)
plot(E_field, P, 'r-', 'LineWidth', 2);   % E in V/m, P in C/m²
hold on;

%%% Extract key parameters
% 1. Positive Remanent Polarization (+Pr) - polarization at E = 0 (upper branch)
E_positive = P > 0;
[~, idx_E0_pos] = min(abs(E_field(E_positive)));
E_pos_indices = find(E_positive);
Pr_pos = P(E_pos_indices(idx_E0_pos));

% 2. Negative Remanent Polarization (-Pr) - polarization at E = 0 (lower branch)
E_negative = P < 0;
[~, idx_E0_neg] = min(abs(E_field(E_negative)));
E_neg_indices = find(E_negative);
Pr_neg = P(E_neg_indices(idx_E0_neg));
Pr = (abs(Pr_pos) + abs(Pr_neg)) / 2; % Average remanent polarization

% 3. Positive Coercive Field (+Ec) - field at P = 0 (right side)
P_near_zero_pos = abs(P) < 0.1*max(abs(P)) & E_field > 0;
if any(P_near_zero_pos)
    indices_pos = find(P_near_zero_pos);
    [~, min_idx] = min(abs(P(indices_pos)));
    Ec_pos = E_field(indices_pos(min_idx));
else
    Ec_pos = NaN;
end

% 4. Negative Coercive Field (-Ec) - field at P = 0 (left side)
P_near_zero_neg = abs(P) < 0.1*max(abs(P)) & E_field < 0;
if any(P_near_zero_neg)
    indices_neg = find(P_near_zero_neg);
    [~, min_idx] = min(abs(P(indices_neg)));
    Ec_neg = E_field(indices_neg(min_idx));
else
    Ec_neg = NaN;
end
Ec = (abs(Ec_pos) + abs(Ec_neg)) / 2;   % Average coercive field

% 5. Maximum (Saturation) Polarization
Ps_pos = max(P);
Ps_neg = min(P);
Ps = (abs(Ps_pos) + abs(Ps_neg)) / 2;

% Find indices for annotations
[~, idx_Ps_pos] = max(P);
[~, idx_Ps_neg] = min(P);
idx_Pr_pos = E_pos_indices(idx_E0_pos);
idx_Pr_neg = E_neg_indices(idx_E0_neg);

% 6. Electric Field at Ps
Es_pos = abs(E_field(idx_Ps_pos));
Es_neg = -abs(E_field(idx_Ps_neg));
Es = (Es_pos - Es_neg) / 2;

%%% Annotate Key Values — all plot calls use E (V/m) and P in C/m²
% Positive Saturation Polarization (+Ps)
h1 = plot(E_field(idx_Ps_pos), P(idx_Ps_pos), 'ks', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'g', 'LineWidth', 1.5);

% Negative Saturation Polarization (-Ps)
plot(E_field(idx_Ps_neg), P(idx_Ps_neg), 'ks', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'g', 'LineWidth', 1.5);

% Positive Remanent Polarization (+Pr)
h2 = plot(0, Pr_pos, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'LineWidth', 1.5);

% Negative Remanent Polarization (-Pr)
plot(0, Pr_neg, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'LineWidth', 1.5);

% Positive Coercive Field (+Ec)
h3 = [];
if ~isnan(Ec_pos)
    h3 = plot(Ec_pos, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
end

% Negative Coercive Field (-Ec)
if ~isnan(Ec_neg)
    plot(Ec_neg, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'LineWidth', 1.5);
end

hold off;

xlabel('Electric Field (V m^{-1})', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Polarization (C m^{-2})', 'FontSize', 12, 'FontWeight', 'bold');
title('Ferroelectric Hysteresis Loop', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xline(0, 'k--', 'LineWidth', 0.8);
yline(0, 'k--', 'LineWidth', 0.8);
box on;

%%% Add legend
if ~isempty(h3)
    legend([h1, h2, h3], ...
        'Saturation Polarization (P_s)', ...
        'Remanent Polarization (P_r)', ...
        'Coercive Field (E_c)', ...
        'Location', 'best', 'FontSize', 10);
else
    legend([h1, h2], ...
        'Saturation Polarization (P_s)', ...
        'Remanent Polarization (P_r)', ...
        'Location', 'best', 'FontSize', 10);
end

%% Print summary to console
fprintf('\n=== Ferroelectric Parameters ===\n');
fprintf('Saturation Polarization (Ps): %.4e C/m²\n', Ps);
fprintf('  at Electric Field (Es): %.4f V/m\n', Es);
fprintf('Remanent Polarization (Pr): %.4e C/m²\n', Pr);
fprintf('Coercive Field (Ec): %.4f V/m\n', Ec);
fprintf('\nDetailed values:\n');
fprintf('  +Ps = %.4e C/m² at +Es = %.4f V/m\n', Ps_pos, Es_pos);
fprintf('  -Ps = %.4e C/m² at -Es = %.4f V/m\n', Ps_neg, Es_neg);
fprintf('  +Pr = %.4e C/m², -Pr = %.4e C/m²\n', Pr_pos, Pr_neg);
fprintf('  +Ec = %.4f V/m, -Ec = %.4f V/m\n', Ec_pos, Ec_neg);