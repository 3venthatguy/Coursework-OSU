clear all; close all; clc;

%% Define frequency bands (normalized by π)
% firls requires frequencies normalized to [0, 1] where 1 corresponds to π
f = [0 0.3 0.4 0.6 0.7 1];  % Frequency band edges (in units of π)
a = [0 0   1   1   0   0];  % Desired amplitude at each band edge

%% Trial and error to find appropriate filter order
% Start with initial guess and increase until stopband attenuation >= 40 dB

% Try different filter orders/coefficients
orders_to_try = [20, 49, 50, 51, 52, 60];

figure('Position', [100, 100, 1200, 800]);

for i = 1:length(orders_to_try)
    n = orders_to_try(i);
    
    % Design filter using firls (least squares method)
    h = firls(n, f, a);
    
    % Compute frequency response
    [H, w] = freqz(h, 1, 2048);
    
    % Convert to dB
    H_dB = 20*log10(abs(H));
    
    % Find minimum stopband attenuation
    stopband1_idx = find(w <= 0.3*pi);
    stopband2_idx = find(w >= 0.7*pi);
    
    max_stopband1 = max(H_dB(stopband1_idx));
    max_stopband2 = max(H_dB(stopband2_idx));
    stopband_atten = max(max_stopband1, max_stopband2);
    
    % Plot in subplot
    subplot(3, 2, i);
    plot(w/pi, H_dB, 'LineWidth', 1.5);
    grid on;
    xlabel('Normalized Frequency (×π rad/sample)');
    ylabel('Magnitude (dB)');
    title(sprintf('Order n = %d, Min Stopband = %.2f dB', n, stopband_atten));
    ylim([-80, 5]);
    
    % Add horizontal line at -40 dB
    hold on;
    yline(-40, 'r--', 'LineWidth', 1.5, 'Label', '-40 dB');
    
    % Highlight passband and stopband regions
    xregion(0, 0.3, 'FaceColor', 'r', 'FaceAlpha', 0.1);
    xregion(0.4, 0.6, 'FaceColor', 'g', 'FaceAlpha', 0.1);
    xregion(0.7, 1, 'FaceColor', 'r', 'FaceAlpha', 0.1);
    
    fprintf('Order n = %d: Stopband attenuation = %.2f dB\n', n, stopband_atten);
end

sgtitle('Filter Order Selection for 40 dB Stopband Attenuation');

%% Select the minimum order that meets specification
% Based on trial and error, select appropriate n
n = 51;  % Adjust this based on your results

fprintf('\n=== Final Design ===\n');
fprintf('Selected filter order: n = %d\n', n);

% Design final filter
h = firls(n, f, a);

% Compute frequency response
[H, w] = freqz(h, 1, 4096);
H_dB = 20*log10(abs(H));
phase = angle(H);
phase_unwrapped = unwrap(phase);

% Calculate stopband attenuation
stopband1_idx = find(w <= 0.3*pi);
stopband2_idx = find(w >= 0.7*pi);
max_stopband1 = max(H_dB(stopband1_idx));
max_stopband2 = max(H_dB(stopband2_idx));
stopband_atten = max(max_stopband1, max_stopband2);

fprintf('Achieved stopband attenuation: %.2f dB\n', stopband_atten);

%% Plot final magnitude response
figure('Position', [100, 100, 1200, 800]);

% Magnitude response in dB
subplot(3, 1, 1);
plot(w/pi, H_dB, 'b', 'LineWidth', 2);
grid on;
xlabel('Normalized Frequency (×π rad/sample)', 'FontSize', 12);
ylabel('Magnitude (dB)', 'FontSize', 12);
title(sprintf('Magnitude Response (Order n = %d)', n), 'FontSize', 14);
ylim([-80, 5]);
xlim([0, 1]);

% Add specification lines
hold on;
yline(-40, 'r--', 'LineWidth', 2, 'Label', '-40 dB Spec');

% Highlight regions
xregion(0, 0.3, 'FaceColor', 'r', 'FaceAlpha', 0.1);
xregion(0.4, 0.6, 'FaceColor', 'g', 'FaceAlpha', 0.1);
xregion(0.7, 1, 'FaceColor', 'r', 'FaceAlpha', 0.1);
legend('Filter Response', '-40 dB', 'Location', 'best');

% Magnitude response (linear scale) - zoomed to passband
subplot(3, 1, 2);
plot(w/pi, abs(H), 'b', 'LineWidth', 2);
grid on;
xlabel('Normalized Frequency (×π rad/sample)', 'FontSize', 12);
ylabel('Magnitude', 'FontSize', 12);
title('Magnitude Response (Linear Scale)', 'FontSize', 14);
xlim([0, 1]);
ylim([0, 1.2]);

% Highlight regions
hold on;
xregion(0, 0.3, 'FaceColor', 'r', 'FaceAlpha', 0.1);
xregion(0.4, 0.6, 'FaceColor', 'g', 'FaceAlpha', 0.1);
xregion(0.7, 1, 'FaceColor', 'r', 'FaceAlpha', 0.1);

% Phase response
subplot(3, 1, 3);
plot(w/pi, phase_unwrapped, 'b', 'LineWidth', 2);
grid on;
xlabel('Normalized Frequency (×π rad/sample)', 'FontSize', 12);
ylabel('Phase (radians)', 'FontSize', 12);
title('Phase Response (Unwrapped)', 'FontSize', 14);
xlim([0, 1]);

% Check for linear phase in passband
passband_idx = find(w >= 0.4*pi & w <= 0.6*pi);
passband_phase = phase_unwrapped(passband_idx);
passband_freq = w(passband_idx);

% Fit line to check linearity
p = polyfit(passband_freq, passband_phase, 1);
hold on;
plot(passband_freq/pi, polyval(p, passband_freq), 'r--', 'LineWidth', 2);
legend('Actual Phase', 'Linear Fit (Passband)', 'Location', 'best');

fprintf('Phase linearity in passband - slope: %.4f rad/sample\n', p(1));
fprintf('Group delay: %.2f samples\n', -p(1));

%% Display filter coefficients
figure('Position', [100, 100, 1000, 400]);
stem(0:n, h, 'LineWidth', 1.5, 'MarkerSize', 4);
grid on;
xlabel('Sample Index n', 'FontSize', 12);
ylabel('Coefficient Value h[n]', 'FontSize', 12);
title(sprintf('FIR Filter Impulse Response (Order n = %d)', n), 'FontSize', 14);

% Check symmetry for linear phase
fprintf('\nChecking filter symmetry (for linear phase):\n');
fprintf('h[0] = %.6f, h[%d] = %.6f\n', h(1), n, h(end));
fprintf('h[1] = %.6f, h[%d] = %.6f\n', h(2), n-1, h(end-1));
fprintf('Filter is symmetric: %d\n', max(abs(h - fliplr(h))) < 1e-10);