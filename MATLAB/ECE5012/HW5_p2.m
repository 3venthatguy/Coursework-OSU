
% Split-step propagators:
%   Diffraction (Fourier space): exp( -(kx^2)/(2*k*n0) * Dz )
%   Refraction  (real space):    exp( (k/(2*n0))*(Er(x)-n0^2) * Dz )
%
% Effective index extracted from variational formula (lecture notes p.4/11):
%   ne^2 = [ k^2*sum(Er*|phi|^2)*Dx - sum(|dphi/dx|^2)*Dx ]
%          / [ k^2 * sum(|phi|^2)*Dx ]

clear; clc; close all;

n1  = 1.60;      
n0  = 1.58;        
a   = 2.0;         
lam = 1.55;        
k   = 2*pi / lam;  

xmin = -20;  xmax = 20;     
N    = 2048;                  % number of points (even)
Dx   = (xmax - xmin) / N;   
x    = (-(N/2) : (N/2)-1) * Dx;   % centred grid

%% Refractive index profile 

n_profile = n0 * ones(1, N);                     
inside     = (abs(x) <= a);
n_profile(inside) = (n1 - n0) * sqrt(a^2 - x(inside).^2) / 1.0 + n0;

Er = n_profile.^2;   

%% Fourier-space wavenumbers (DFT convention, centred) 
m  = (-(N/2) : (N/2)-1);
kx = (2*pi * m) / (N * Dx);   % [1/um]

%% Propagation parameters 
Dz      = 0.05;       % imaginary-distance step 
z_max   = 800;        % max propagation
n_steps = round(z_max / Dz);

%% Pre-compute split-step propagator factors 
% Diffraction: acts in Fourier space
prop_A = exp( -(kx.^2) / (2*k*n0) * Dz );  

% Refraction: acts in real space
prop_B = exp( (k / (2*n0)) * (Er - n0^2) * Dz );  % real, 1xN

%% Initial condition 
sigma0 = a;
phi    = exp(-x.^2 / (2*sigma0^2));

%% Imaginary-distance BPM loop 
ne_prev = 0;
tol     = 1e-8;     
ne_hist = nan(1, n_steps);

converged_step = n_steps;  

for step = 1 : n_steps

    phi = prop_B .* phi;

    % Use centred FFT: fftshift/ifftshift to match DFT convention in notes
    PHI = fftshift( fft( ifftshift(phi) ) ) / N;
    PHI = prop_A .* PHI;
    phi = fftshift( ifft( ifftshift(PHI) ) ) * N;

    % Keep phi real (small imaginary parts accumulate from floating point)
    phi = real(phi);

    % Normalise to prevent overflow (arbitrary scale; ne is scale-free)
    phi = phi / max(abs(phi));

    % Compute ne using variational formula 
    dphi_dx = gradient(phi, Dx);      
    num = k^2 * sum(Er .* phi.^2) * Dx  -  sum(dphi_dx.^2) * Dx;
    den = k^2 * sum(phi.^2) * Dx;
    ne2 = num / den;
    ne  = sqrt(max(real(ne2), 0));
    ne_hist(step) = ne;

    % Convergence check
    if step > 10 && abs(ne - ne_prev) < tol
        converged_step = step;
        fprintf('Converged at step %d,  z = %.2f um\n', step, step*Dz);
        break;
    end
    ne_prev = ne;
end

z_final = converged_step * Dz;

%% Report 
fprintf('\nNumerical  ne = %.4f\n', ne);
fprintf('Dx = %.4f um,  Dz = %.4f um,  z_final = %.4f um\n', Dx, Dz, z_final);

phi_norm = phi / max(abs(phi));   % normalise to maximum = 1

figure('Color','w', 'Position', [100 100 900 520]);
plot(x, phi_norm, 'b-', 'LineWidth', 2);
xlim([-20  20]);
ylim([ 0    1]);
xlabel('x = n\Deltax (\mum)',  'FontSize', 13);
ylabel('\phi(n) (norm.)',       'FontSize', 13);
title(sprintf(['Numerical n_e = %.4f,  \\Deltax = %.4f (\\mum),  ' ...
               '\\Deltaz = %.4f (\\mum),  z = %.4f (\\mum)'], ...
    ne, Dx, Dz, z_final), 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12);