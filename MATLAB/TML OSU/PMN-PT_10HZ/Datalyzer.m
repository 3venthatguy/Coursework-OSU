clear; clc; close all;
data = readmatrix("600.csv");

% Extract time and signal columns from the data
time = data(:, 1);
CH1 = data(:, 2) * 10;
CH2 = data(:, 3);

% ========== Calculate Critical Values ==========

% 1. Coercive Field (Ec) ========================

% Separate into positive and negative half-cycles
mid_point = round(length(CH1_actual)/2);

% Positive to negative transition (left coercive field)
[~, idx_left] = min(abs(CH2(1:mid_point)));
Ec_left = CH1_actual(idx_left);

% Negative to positive transition (right coercive field)
[~, idx_right] = min(abs(CH2(mid_point:end)));
idx_right = idx_right + mid_point - 1;
Ec_right = CH1_actual(idx_right);

Ec_avg = (abs(Ec_left) + abs(Ec_right)) / 2;

% 2. Remanent Polarization (Pr) =================
% Upper remanent polarization
[~, idx_upper] = min(abs(CH1_actual(1:mid_point)));
Pr_upper = CH2(idx_upper);

% Lower remanent polarization
[~, idx_lower] = min(abs(CH1_actual(mid_point:end)));
idx_lower = idx_lower + mid_point - 1;
Pr_lower = CH2(idx_lower);

Pr_avg = (abs(Pr_upper) + abs(Pr_lower)) / 2;

% Plot the signals from CH1 and CH2
figure(1);
plot(CH1, CH2, 'b-', 'LineWidth', 1.5);
xlabel('CH1 - Applied Field (V)');
ylabel('CH2 - Response (V)');
title('Hysteresis Loop');
grid on;
axis tight;

figure(2);
% Plot the time series for CH1 and CH2
plot(time, CH1, 'r-', 'LineWidth', 1.5);
hold on;
plot(time, CH2, 'g-', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Signal Amplitude (V)');
title('Time Series of CH1 and CH2');
legend('CH1', 'CH2');
grid on;
hold off;


% ==================== FUNCTIONS ====================

function [Ec_right, Ec_left, Ec_avg]