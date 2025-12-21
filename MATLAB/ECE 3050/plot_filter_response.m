function plot_filter_response(Lh, h1, N_freq)

[H1, w] = freqz(h1, 1, N_freq);
f_normalized = w/pi;

figure('Position', [100 100 1200 800]);

% Subplot 1: Magnitude Response (Linear scale)
subplot(2,2,1);
plot(f_normalized, abs(H1), 'b', 'LineWidth', 1.5);
hold on;
% Mark specification bands
xline(0.15, 'r--', 'LineWidth', 1.5, 'Label', 'Passband edge');
xline(0.25, 'k--', 'LineWidth', 1.5, 'Label', 'Stopband edge');
fill([0.15 0.25 0.25 0.15], [0 0 1.2 1.2], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text(0.2, 0.6, 'Transition', 'FontSize', 10, 'HorizontalAlignment', 'center');
hold off;
grid on;
xlabel('Normalized Frequency (×π rad/sample)');
ylabel('Magnitude');
title('Magnitude Response (Linear Scale)');
xlim([0 1]);
ylim([0 1.2]);
legend('|H_1(e^{j\omega})|', 'Location', 'best');

% Subplot 2: Magnitude Response (dB scale)
subplot(2,2,2);
plot(f_normalized, 20*log10(abs(H1)), 'b', 'LineWidth', 1.5);
hold on;
xline(0.15, 'r--', 'LineWidth', 1.5);
xline(0.25, 'k--', 'LineWidth', 1.5);
fill([0.15 0.25 0.25 0.15], [-100 -100 5 5], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
hold off;
grid on;
xlabel('Normalized Frequency (×π rad/sample)');
ylabel('Magnitude (dB)');
title('Magnitude Response (dB Scale)');
xlim([0 1]);
ylim([-100 5]);

% Subplot 3: Phase Response
subplot(2,2,3);
plot(f_normalized, angle(H1), 'b', 'LineWidth', 1.5);
hold on;
xline(0.15, 'r--', 'LineWidth', 1.5);
xline(0.25, 'k--', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('Normalized Frequency (×π rad/sample)');
ylabel('Phase (radians)');
title('Phase Response');
xlim([0 1]);

% Subplot 4: Impulse Response
subplot(2,2,4);
stem(0:Lh-1, h1, 'b', 'filled', 'LineWidth', 1.5);
grid on;
xlabel('Sample index n');
ylabel('h_1[n]');
title('Impulse Response (Filter Coefficients)');

end