clear; clc; close all;

%{
%% Load and Prepare Data
data = readmatrix("C:\Users\evanm\Downloads\TempDownloads\260.csv");
% data = readmatrix("/Users/evansmacbookair/Downloads/TempDownloads/260.csv");

time = data(:, 1);
CH1 = data(:, 2) * 10;  % Account for 10x amplifier
CH2 = data(:, 3);
%}

%% Load Both Datasets

data_pos = readmatrix("C:\Users\evanm\Downloads\TempDownloads\260.csv");
data_neg = readmatrix("C:\Users\evanm\Downloads\TempDownloads\260.csv");

% Extract positive half data
time_pos = data_pos(:, 1);
CH1_pos = data_pos(:, 2) * 10;  % Account for 10x amplifier
CH2_pos = data_pos(:, 3);

% Keep only positive voltages (upper half of hysteresis)
valid_pos = CH1_pos > 0;
time_pos = time_pos(valid_pos);
CH1_pos = CH1_pos(valid_pos);
CH2_pos = CH2_pos(valid_pos);

% Extract negative half data
time_neg = data_neg(:, 1);
CH1_neg = data_neg(:, 2) * 10;  % Account for 10x amplifier
CH2_neg = data_neg(:, 3);

% Keep only negative voltages (lower half of hysteresis)
valid_neg = CH1_neg < 0;
time_neg = time_neg(valid_neg);
CH1_neg = CH1_neg(valid_neg);
CH2_neg = CH2_neg(valid_neg);

% Combine both halves
CH1 = [CH1_pos; CH1_neg];
CH2 = [CH2_pos; CH2_neg];

% Sort by CH1 to form proper hysteresis loop
[CH1, sort_idx] = sort(CH1);
CH2 = CH2(sort_idx);

%% Calculate Critical Values

% Split data into two half-cycles
% mid_point = round(length(CH1) / 2);
[~, mid_point] = min(abs(diff(sign(CH1))));

% === Coercive Field (Ec) - where CH2 crosses zero ===
[~, idx_left] = min(abs(CH2(1:mid_point)));
Ec_left = CH1(idx_left);

[~, idx_right] = min(abs(CH2(mid_point:end)));
idx_right = idx_right + mid_point - 1;
Ec_right = CH1(idx_right);

Ec_avg = (abs(Ec_left) + abs(Ec_right)) / 2;

% === Remanent Polarization (Pr) - where CH1 crosses zero ===
[~, idx_upper] = min(abs(CH1(1:mid_point)));
Pr_upper = CH2(idx_upper);

[~, idx_lower] = min(abs(CH1(mid_point:end)));
idx_lower = idx_lower + mid_point - 1;
Pr_lower = CH2(idx_lower);

Pr_avg = (abs(Pr_upper) + abs(Pr_lower)) / 2;

% === Maximum Polarization (Pmax) - peak values ===
[Pmax_pos, idx_Pmax_pos] = max(CH2);
[Pmax_neg, idx_Pmax_neg] = min(CH2);
Pmax_avg = (abs(Pmax_pos) + abs(Pmax_neg)) / 2;

E_at_Pmax_pos = CH1(idx_Pmax_pos);
E_at_Pmax_neg = CH1(idx_Pmax_neg);

%% Display Results
fprintf('\n========== Critical Values ==========\n\n');
fprintf('Coercive Field (Ec):\n');
fprintf('  Left:   %8.2f V\n', Ec_left);
fprintf('  Right:  %8.2f V\n', Ec_right);
fprintf('  Average:%8.2f V\n\n', Ec_avg);

fprintf('Remanent Polarization (Pr):\n');
fprintf('  Upper:  %8.4f V\n', Pr_upper);
fprintf('  Lower:  %8.4f V\n', Pr_lower);
fprintf('  Average:%8.4f V\n\n', Pr_avg);

fprintf('Maximum Polarization (Pmax) - NOT SATURATED:\n');
fprintf('  Positive: %8.4f V at E = %6.2f V\n', Pmax_pos, E_at_Pmax_pos);
fprintf('  Negative: %8.4f V at E = %6.2f V\n', Pmax_neg, E_at_Pmax_neg);
fprintf('  Average:  %8.4f V\n', Pmax_avg);
fprintf('\n======================================\n\n');

%% Figure 1: Hysteresis Loop with Critical Points
figure(1);
plot(CH1, CH2, 'b-', 'LineWidth', 2);
hold on;

% Mark critical points
plot([Ec_left, Ec_right], [0, 0], 'ro', 'MarkerSize', 10, ...
     'LineWidth', 2, 'MarkerFaceColor', 'r');
plot([0, 0], [Pr_upper, Pr_lower], 'gs', 'MarkerSize', 10, ...
     'LineWidth', 2, 'MarkerFaceColor', 'g');
plot([E_at_Pmax_pos, E_at_Pmax_neg], [Pmax_pos, Pmax_neg], 'md', ...
     'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'm');

% Reference lines
xline(0, 'k--', 'LineWidth', 0.75);
yline(0, 'k--', 'LineWidth', 0.75);

xlabel('CH1 - Applied Field (V)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CH2 - Polarization Response (V)', 'FontSize', 12, 'FontWeight', 'bold');
title('PMN Ferroelectric Hysteresis Loop', 'FontSize', 14, 'FontWeight', 'bold');

legend('Hysteresis Loop', 'Coercive Field (E_c)', ...
       'Remanent Polarization (P_r)', 'Maximum Polarization (P_{max})', ...
       'Location', 'southeast', 'FontSize', 9);

grid on;
xlim([min(CH1)*1.30, max(CH1)*1.30]);
ylim([min(CH2)*1.30, max(CH2)*1.30]);
set(gca, 'FontSize', 11);
hold off;

%% Other Figures
figure(2);
plot(time_pos, CH1_pos, 'r-', 'LineWidth', 2);
hold on;
plot(time_neg, CH1_neg, 'g-', 'LineWidth', 2);

%{
%% Figure 2: Time Series of CH1 and CH2
figure(2);
plot(time, CH1, 'r-', 'LineWidth', 2);
hold on;
plot(time, CH2, 'g-', 'LineWidth', 2);

xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Signal Amplitude (V)', 'FontSize', 12, 'FontWeight', 'bold');
title('Time Series: Applied Field and Polarization Response', ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('CH1 - Applied Field (10x)', 'CH2 - Polarization', ...
       'Location', 'best', 'FontSize', 10);

grid on;
set(gca, 'FontSize', 11);
hold off;

%% Figure 3: Time Series of CH2 Only
figure(3);
plot(time, CH2, 'g-', 'LineWidth', 2);

xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CH2 - Polarization Response (V)', 'FontSize', 12, 'FontWeight', 'bold');
title('Time Series: Polarization Response', 'FontSize', 14, 'FontWeight', 'bold');

grid on;
set(gca, 'FontSize', 11);
%}