%% Q1

rng_seed = 3050;
[x, Lx] = gen_input(rng_seed);

%% Q2

% Filter Specifications
Lh = 11;
order = Lh - 1;
N_freq = 1024;

% Frequency bands (normalized frequency, where 1.0 = pi rad/sample)
f_pass = [0, 0.15];
f_trans = [0.15, 0.25];
f_stop = [0.25, 1.0];

% frils requirses band edges and desired amplitudes for LOW PASS
freq_edges_1 = [f_pass(1), f_pass(2), f_stop(1), f_stop(2)];
desired_amp_1 = [1, 1, 0, 0];     % 1 in passband, 0 in stopband
h1 = firls(order, freq_edges_1, desired_amp_1);

plot_filter_response(Lh, h1, N_freq);

%% Q3

f_stop_2 = [0, 0.15];
f_trans_2 = [0.15, 0.25];
f_pass_2 = [0.25, 1.0];

% New frils requirses band edges and desired amplitudes for HIGH PASS
freq_edges_2 = [f_stop_2(1), f_stop_2(2), f_pass_2(1), f_pass_2(2)];
desired_amp_2 = [0, 0, 1, 1];     % 1 in passband, 0 in stopband
h2 = firls(order, freq_edges_2, desired_amp_2);

plot_filter_response(Lh, h2, N_freq);

%% Q4

y1 = conv(h1, x);
y2 = conv(h2, x);
Ly = length(y1);

figure();

fft_size = 2^ceil(log2(Ly));
f = linspace(0, 1, fft_size/2);
fft_x = fft(x, fft_size);
fft_y1 = fft(y1, fft_size);
fft_y2 = fft(y2, fft_size);

subplot(3,1,1)
plot(f, abs(fft_x(1:fft_size/2)), 'r-','LineWidth',1); hold on; grid on;
title('FFT Magnitude of x');
xlabel('Normalized Frequency'); ylabel('Magnitude');

subplot(3,1,2)
plot(f, abs(fft_y1(1:fft_size/2)), 'r-','LineWidth',1); hold on; grid on;
title('FFT Magnitude of y1');
xlabel('Normalized Frequency'); ylabel('Magnitude');

subplot(3,1,3)
plot(f, abs(fft_y2(1:fft_size/2)), 'r-','LineWidth',1); hold on; grid on;
title('FFT Magnitude of y2');
xlabel('Normalized Frequency'); ylabel('Magnitude');

%% Q5

Y1 = convmtx_lin(y1, Lh);
Y2 = convmtx_lin(y2, Lh);

% Construct matrix
A = [Y2, -Y1];

% Find null vector using SVD
[U, S, V] = svd(A, 'econ');
v = V(:, end);

% Extract estimated filter vectors
h1_est = v(1:Lh);
h2_est = v(Lh+1:end);

% Scale estimated filters to match original filter norms
scale_factor = norm(h1) / norm(h1_est);
h1_est_scaled = h1_est * scale_factor;
h2_est_scaled = h2_est * scale_factor;

%% Q6

H1_est = convmtx_lin(h1_est_scaled, Lx);
H2_est = convmtx_lin(h2_est_scaled, Lx);

H_stack = [H1_est; H2_est];
y_stack = [y1; y2];
x_est = mldivide(H_stack, y_stack);

rho = (x' * x_est) / norm(x) / norm(x_est);

fprintf('\n=== Signal Recovery Results ===\n');
fprintf('Correlation coefficient (rho): %.6f\n', rho);
fprintf('Absolute correlation: %.6f\n', abs(rho));
fprintf('Angle between vectors: %.2f degrees\n', acosd(rho));

%% Functions

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

function Y = convmtx_lin(x, n)
x = x(:); m = length(x); L = m + n - 1;
Y = zeros(L, n);
for k = 1:n
    Y(k:k+m-1, k) = x;
end
end