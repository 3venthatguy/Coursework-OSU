clear; clc; close all;

% Given parameters
n1 = 2.942;  
ns = 1.460;  
no = 1.460;  

% Normalized frequency range
v = linspace(0.01, 5, 1000);

% Initialize modes
m = [0, 1, 2];
b_TM = zeros(length(m), length(v));

% Cutoff frequencies
v_cutoff = m * pi / 2;

%% Solve for TM modes
for mode_idx = 1:length(m)
    for i = 1:length(v)
        dispersion_eq = @(b) 2*v(i)*sqrt(1-b) - atan((n1^2/ns^2)*sqrt(b./(1-b))) - atan((n1^2/no^2)*sqrt(b./(1-b))) - m(mode_idx)*pi;

        try
            b_TM(mode_idx, i) = fzero(dispersion_eq, [0.001, 0.999]);
        catch
            b_TM(mode_idx, i) = NaN;
        end
    end
end

%% Plot the dispersion curves
figure('Position', [100, 100, 800, 600]);
hold on; grid on; box on;

% Colors for each mode
colors = {'b', 'r', 'g'};
markers = {'o', 'o', 'o'};

% Plot each mode
for mode_idx = 1:length(m)
    % Only plot valid values (above cutoff and within [0,1])
    valid = ~isnan(b_TM(mode_idx, :)) & ...
            b_TM(mode_idx, :) >= 0 & ...
            b_TM(mode_idx, :) <= 1 & ...
            v >= v_cutoff(mode_idx);
    
    plot(v(valid), b_TM(mode_idx, valid), ...
         [colors{mode_idx} '-'], 'LineWidth', 2, ...
         'DisplayName', sprintf('TM_%d', m(mode_idx)));
    
    % Add cutoff marker (except for TM0)
    if m(mode_idx) > 0
        plot(v_cutoff(mode_idx), 0, [colors{mode_idx} markers{mode_idx}], ...
             'MarkerSize', 8, 'MarkerFaceColor', colors{mode_idx}, ...
             'HandleVisibility', 'off');
    end
end

% Labels and formatting
xlabel('Normalized Frequency, v', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Normalized Propagation Constant, b', 'FontSize', 14, 'FontWeight', 'bold');
title(sprintf('TM Mode Dispersion Relations (n_1 = %.3f, n_s = n_o = %.3f)', n1, ns), ...
    'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 12);

% Set axis limits
xlim([0, 5]);
ylim([0, 1]);

% Grid formatting
grid on;
set(gca, 'FontSize', 12);

hold off;