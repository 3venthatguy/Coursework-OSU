clear; clc; close all;

% Constants
lambda = 1.55;
a = 0.225;
n1 = 2.31;
n0 = 1.46;
Kl = 0.14321;

% Solve waveguide parameters
k = 2*pi/lambda;
v = k*a*sqrt(n1^2 - n0^2);
eq = @(b) v*sqrt(1-b) - atan(sqrt(b./(1-b)));
b = fzero(eq, [0, 0.999]);

ne = sqrt(n0^2 + b*(n1^2 - n0^2));
beta = k*ne;
kappa = sqrt(k^2*n1^2 - beta^2);
sigma = sqrt(beta^2 - k^2*n0^2);

% R >= 3um -- this constraint is met at minimum value m=26
R = lambda*26 / (2*pi*ne);
L = R*2*pi;

% Marcus Approximation
alpha = 0.5*(sigma^2 / (beta*(sigma*a+1)) * kappa^2 / (k^2*(n1^2-n0^2)) ...
    * exp(2*sigma*a) * exp(-2*beta*R*atanh(sigma/beta) + 2*R*sigma));

%% Transmission

% Given definitions for Transmission
x = exp(-alpha*L);
y = cos(Kl);

% Wavelength array
wave_range = linspace(1.54, 1.56, 1000);

% Parameters for each vector in array
k_vec = 2*pi./wave_range;
beta_vec = ne.*k_vec;
phi_vec = beta_vec*L;

% Transmission
T = 1 -((1-x^2)*(1-y^2) ./ ((1-x*y)^2 + 4*x*y * sin(phi_vec./2).^2));

% Plot
figure;
plot(wave_range, T, 'b-', 'LineWidth', 1.5);
xlabel('Wavelength \lambda (\mum)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Transmission T', 'FontSize', 12, 'FontWeight', 'bold');
title('Ring Resonator Filter', 'FontSize', 14);
grid on;
axis tight;