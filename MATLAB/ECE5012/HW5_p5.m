clear; clc; close all;

a = 0.22;
R = 2.8876;
L = R*2*pi;
n0 = 1;
n1 = 2;
wave_range = linspace(1.54, 1.56, 1000);

% Parameters
y = 0.98;               % y = cos(kl)
z = sin(acos(y));       % z = sin(kl)

% Solve waveguide at each wavelength
N = length(wave_range);
beta_vec = zeros(1, N);
alpha_vec = zeros(1, N);

for i = 1:N
    lambda = wave_range(i);
    k = 2*pi/lambda;
    v = k*a*sqrt(n1^2 - n0^2);
    eq = @(b) v*sqrt(1-b) - atan(sqrt(b./(1-b)));
    b = fzero(eq, [0, 0.999]);
    ne = sqrt(n0^2 + b*(n1^2 - n0^2));
    beta = k*ne;
    beta_vec(i) = beta;
    kappa = sqrt(k^2*n1^2 - beta^2);
    sigma = sqrt(beta^2 - k^2*n0^2);

    % Marcus Approx
    alpha = 0.5*(sigma^2 / (beta*(sigma*a+1)) * kappa^2 / (k^2*(n1^2-n0^2)) ...
        * exp(2*sigma*a) * exp(-2*beta*R*atanh(sigma/beta) + 2*R*sigma));
    alpha_vec(i) = alpha;
end

T_drop = zeros(1, N);
T_thru = zeros(1, N);

for i = 1:N
    Delta = exp(-1j*beta_vec(i)*L/2) * exp(-alpha_vec(i)*L/2);
    denominator = 1- y^2*Delta^2;

    T_drop(i) = z^4 * abs(Delta)^2 / abs(denominator)^2;
    T_thru(i) = y^2 * abs(1 - Delta^2)^2 / abs(denominator)^2;
end

figure;
plot(wave_range, T_drop, 'b-', 'LineWidth', 1.5); hold on;
plot(wave_range, T_thru, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength \lambda (\mum)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Transmission T', 'FontSize', 12, 'FontWeight', 'bold');
legend('T_{drop}', 'T_{thru}', 'Location', 'best', 'FontSize', 11);
title('Dual Bus  Coupled Ring Resonator', 'FontSize', 14);
grid on;
axis tight;