clear; clc; close all;

%% Parameters
lambda = 1.55;          % um
n1 = 2.31;              % core index
n2 = 1.46;              % cladding index
a = 0.225;              % half-width um (2a = 0.45um)
D = 0.65;               % gap separation um

%% Waveguide parameters
k = 2*pi/lambda;
v = k*a*sqrt(n1^2 - n2^2);

% Solve TE0 dispersion for b
eq = @(b) v*sqrt(1-b) - atan(sqrt(b./(1-b)));
b = fzero(eq, [0, 0.999]);

ne = sqrt(n2^2 + b*(n1^2 - n2^2));
beta = k*ne;
K_wg = sqrt(k^2*n1^2 - beta^2);   % transverse wavenumber in core
Sigma = sqrt(beta^2 - k^2*n2^2);  % evanescent decay in cladding

% Normalized parameters
u = K_wg * a;
w = Sigma * a;

fprintf('V = %.4f\n', v);
fprintf('b = %.4f\n', b);
fprintf('ne = %.4f\n', ne);
fprintf('beta = %.4f 1/um\n', beta);
fprintf('K = %.4f 1/um\n', K_wg);
fprintf('Sigma = %.4f 1/um\n', Sigma);
fprintf('u = %.4f\n', u);
fprintf('w = %.4f\n', w);

%% Numerator - integration by parts
% Integrand: cos(K*x) * cos(K*a) * exp(-Sigma*(x - D + a))
% cos(Ka) and exp terms evaluated at boundary, pulled out
% Remaining integral: integral of cos(K*x)*exp(-Sigma*x) from -a to a

% Analytical result of integral by parts:
% integral cos(Kx)*exp(-Sigma*x)dx = 
% exp(-Sigma*x)*(−Sigma*cos(Kx) + K*sin(Kx)) / (Sigma^2 + K^2)

denom_int = Sigma^2 + K_wg^2;
upper = exp(-Sigma*a) * (-Sigma*cos(K_wg*a) + K_wg*sin(K_wg*a)) / denom_int;
lower = exp( Sigma*a) * (-Sigma*cos(K_wg*a) - K_wg*sin(K_wg*a)) / denom_int;
core_integral = upper - lower;

% Full numerator prefactor for TM
% (beta/(w*eps0*n1*n2))^2 but omega*eps0 cancels partially
% From notes: prefactor is (beta/(omega*eps0*n1*n2))^2
% omega*eps0 out front cancels one power leaving beta^2/(omega*eps0*n1^2*n2^2)

prefactor_num = (beta^2 / (n1^2 * n2^2));
K_num = prefactor_num * (n1^2 - n2^2) * cos(K_wg*a) * exp(Sigma*(D-a)) * core_integral;

%% Denominator from notes
% K_den = (2*beta*a / (omega*eps0*n1^2)) * 
%         (1 + (2*n2^2/n1^4)*(sin^2(u)/(2w)) + (2/n2^2)*(cos^2(u)/(2w)))

K_den = (2*a / n1^2) * ...
        (1 + (2*n2^2/n1^4)*(sin(u)^2/(2*w)) + (2/n2^2)*(cos(u)^2/(2*w)));

%% K12
K12 = K_num / K_den;

fprintf('\nK12 = %.6f 1/um\n', K12);