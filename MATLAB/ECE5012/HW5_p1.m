%   d(phi)/d(zeta) = [1/(2*k*n0)] * d^2(phi)/dx^2
%                  + [k/(2*n0)] * (Er(x) - n0^2) * phi

% After convergence phi -> fundamental mode, and ne is extracted from:
%
%   ne^2 = [ k^2 * int(Er*|phi|^2 dx) - int(|dphi/dx|^2 dx) ]
%          / [ k^2 * int(|phi|^2 dx) ]

clear; clc; close all;

n1  = 1.60;    
n0  = 1.58;       
a   = 1.0;         
lam = 1.55;         
k   = 2*pi/lam;      

xmin = -20;   xmax = 20;    
N    = 2048;                 % number of transverse points (even)
Dx   = (xmax - xmin) / N;   % transverse step
x    = (-(N/2) : (N/2)-1) * Dx;   % transverse coordinate 

% Propagation parameters
Dz       = 0.05;    % imaginary-distance step
z_max    = 500;     % total propagation distance 
n_steps  = round(z_max / Dz);

%% Refractive index profile
Er = n0^2 * ones(1, N);        % cladding everywhere
Er(abs(x) <= a) = n1^2;        % core

%% Fourier-space wavenumbers (DFT convention, centred)
% m runs from -N/2 to N/2-1
m   = (-(N/2) : (N/2)-1);
kx  = (2*pi*m) / (N*Dx);      

%% Split-step propagator factors (computed once)
% Diffraction:  exp(A_hat * Dz) in Fourier space
prop_A = exp( -(kx.^2) / (2*k*n0) * Dz );  

% Refraction:  exp(B_hat * Dz) in real space
%   B_hat = (k/(2*n0)) * (Er - n0^2)
prop_B = exp( (k/(2*n0)) * (Er - n0^2) * Dz );  

%% Initial condition 
% Any smooth function works; a Gaussian centred on core is convenient
sigma0 = a;
phi = exp(-x.^2 / (2*sigma0^2));   % Gaussian

%% Imaginary-distance BPM loop
ne_prev = 0;
tol     = 1e-8;       % convergence criterion on ne
ne_hist = zeros(1, n_steps);

for step = 1:n_steps
    phi = prop_B .* phi;   
                           
    PHI = fftshift( fft(ifftshift(phi)) ) / N;   % DFT (centred)
    PHI = prop_A .* PHI;
    phi = fftshift( ifft(ifftshift(PHI)) ) * N;  % IDFT (centred)

    % Normalise to prevent overflow
    phi = phi / max(abs(phi));

    % Compute ne after each step
    dphi_dx = gradient(phi, Dx);                 
    num = k^2 * sum(Er .* abs(phi).^2) * Dx ...
        - sum(abs(dphi_dx).^2) * Dx;
    den = k^2 * sum(abs(phi).^2) * Dx;
    ne2 = num / den;
    ne  = sqrt(real(ne2));
    ne_hist(step) = ne;

    % Check convergence
    if abs(ne - ne_prev) < tol && step > 10
        fprintf('Converged at step %d,  z = %.2f um\n', step, step*Dz);
        break;
    end
    ne_prev = ne;
end

z_final = step * Dz;
fprintf('\nNumerical  ne = %.4f\n', ne);

%% Analytical TE dispersion relation 
% TE even modes:  kappa*tan(kappa*a) = gamma
%   kappa^2 = k^2*(n1^2 - ne^2),  gamma^2 = k^2*(ne^2 - n0^2)
%   Solve for ne in [n0, n1]

ne_TE = solve_dispersion(k, n1, n0, a, 'TE');
ne_TM = solve_dispersion(k, n1, n0, a, 'TM');

fprintf('Analytical TE ne = %.4f\n', ne_TE);
fprintf('Analytical TM ne = %.4f\n', ne_TM);

%% Plot 
phi_norm = abs(phi) / max(abs(phi));

figure('Color','w','Position',[100 100 800 500]);
plot(x, phi_norm, 'b-', 'LineWidth', 2);
xlim([-20 20]);
ylim([0 1]);
xlabel('x = n\Deltax (\mum)', 'FontSize', 13);
ylabel('\phi(n) (norm.)',     'FontSize', 13);
title(sprintf(['Numerical n_e = %.4f,  TE n_e = %.4f,  TM n_e = %.4f,  ' ...
               '\Deltax = %.4f (\\mum),  \Deltaz = %.4f (\\mum),  z = %.4f (\\mum)'], ...
    ne, ne_TE, ne_TM, Dx, Dz, z_final), 'FontSize', 10);
grid on;
set(gca,'FontSize',12);

%% 
function ne_out = solve_dispersion(k, n1, n0, a, pol)
    ne_vec = linspace(n0+1e-9, n1-1e-9, 1e6);
    kappa  = k * sqrt(n1^2 - ne_vec.^2);
    gamma  = k * sqrt(ne_vec.^2 - n0^2);

    if strcmpi(pol, 'TE')
        lhs = kappa .* tan(kappa * a);
        rhs = gamma;
    else  % TM
        lhs = kappa .* tan(kappa * a);
        rhs = (n1/n0)^2 * gamma;
    end

    f = lhs - rhs;

    % Find zero crossings (sign changes) – pick the first one (fundamental)
    idx = find(diff(sign(f)) < 0, 1, 'first');  % negative crossing = proper root
    if isempty(idx)
        warning('No root found for %s; returning NaN', pol);
        ne_out = NaN;
        return;
    end

    % Refine with bisection between idx and idx+1
    ne_lo = ne_vec(idx);
    ne_hi = ne_vec(idx+1);
    for iter = 1:60
        ne_mid = (ne_lo + ne_hi) / 2;
        kap    = k * sqrt(n1^2 - ne_mid^2);
        gam    = k * sqrt(ne_mid^2 - n0^2);
        if strcmpi(pol, 'TE')
            fmid = kap * tan(kap*a) - gam;
        else
            fmid = kap * tan(kap*a) - (n1/n0)^2 * gam;
        end
        if fmid * (k*sqrt(n1^2-ne_lo^2)*tan(k*sqrt(n1^2-ne_lo^2)*a) - ...
                   (strcmpi(pol,'TM')*(n1/n0)^2 + strcmpi(pol,'TE'))*k*sqrt(ne_lo^2-n0^2)) < 0
            ne_hi = ne_mid;
        else
            ne_lo = ne_mid;
        end
    end
    ne_out = (ne_lo + ne_hi) / 2;
end