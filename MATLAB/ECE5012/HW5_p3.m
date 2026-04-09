%% BPM_higher_order_mode.m
% Problem 3: Scalar imaginary-distance BPM for the FIRST HIGHER-ORDER (m=1)
% mode of a slab waveguide.
%
% Parameters: n1=1.60, n0=1.58, a=3 um, lambda=1.55 um
%
% Strategy (from lecture notes):
%   The imaginary-distance BPM naturally converges to the mode with the
%   LARGEST ne (fundamental, m=0).  To isolate the m=1 mode we exploit
%   orthogonality:  at each propagation step, project out the m=0
%   component using
%
%       phi <- phi - [<phi0, phi> / <phi0, phi0>] * phi0
%
%   where phi0 is the (previously computed) fundamental mode and <f,g>
%   denotes the inner product  sum(f .* g) * Dx.
%   After deflation the BPM converges to the next-largest-ne mode (m=1).
%
% The m=1 mode is ODD (antisymmetric), so its field distribution
% has a node at x=0 and lobes of opposite sign — hence y-axis [-1, +1].
%
% Effective index formula (lecture notes p.4/11):
%   ne^2 = [k^2*int(Er*|phi|^2 dx) - int(|dphi/dx|^2 dx)]
%          / [k^2 * int(|phi|^2 dx)]
%
% Analytical TE/TM dispersion relations for ODD modes:
%   TE:  -kappa*cot(kappa*a) = gamma
%   TM:  -kappa*cot(kappa*a) = (n1/n0)^2 * gamma
%   kappa = k*sqrt(n1^2 - ne^2),  gamma = k*sqrt(ne^2 - n0^2)

clear; clc; close all;

%% ===== Physical parameters =====
n1  = 1.60;
n0  = 1.58;
a   = 3.0;           % half-width [um]
lam = 1.55;          % free-space wavelength [um]
k   = 2*pi / lam;    % free-space wavenumber [1/um]

%% ===== Transverse grid =====
xmin = -40;  xmax = 40;      % [um]  (wider window for a=3 um)
N    = 4096;                  % even number of points
Dx   = (xmax - xmin) / N;    % [um]
x    = (-(N/2) : (N/2)-1) * Dx;   % centred grid [um]

%% ===== Refractive index profile (step-index slab) =====
Er          = n0^2 * ones(1, N);
Er(abs(x) <= a) = n1^2;

%% ===== Fourier-space wavenumbers =====
m  = (-(N/2) : (N/2)-1);
kx = (2*pi * m) / (N * Dx);   % [1/um]

%% ===== Propagation parameters =====
Dz      = 0.05;        % imaginary-distance step [um]
z_max   = 1000;        % max propagation [um]
n_steps = round(z_max / Dz);

%% ===== Pre-compute split-step propagators =====
prop_A = exp( -(kx.^2) / (2*k*n0) * Dz );          % diffraction (k-space)
prop_B = exp( (k/(2*n0)) * (Er - n0^2) * Dz );      % refraction  (x-space)

%% =====================================================================
%% STEP 1: Compute the fundamental mode (m=0) first
%% =====================================================================
fprintf('=== Computing fundamental mode (m=0) ===\n');

sigma0 = a;
phi0   = exp(-x.^2 / (2*sigma0^2));

ne0_prev = 0;  tol = 1e-8;  ne0 = 0;
converged_step0 = n_steps;

for step = 1:n_steps
    phi0 = prop_B .* phi0;
    PHI0 = fftshift(fft(ifftshift(phi0))) / N;
    PHI0 = prop_A .* PHI0;
    phi0 = real(fftshift(ifft(ifftshift(PHI0))) * N);
    phi0 = phi0 / max(abs(phi0));

    dphi0   = gradient(phi0, Dx);
    num0    = k^2*sum(Er.*phi0.^2)*Dx - sum(dphi0.^2)*Dx;
    den0    = k^2*sum(phi0.^2)*Dx;
    ne0     = sqrt(max(real(num0/den0), 0));

    if step > 10 && abs(ne0 - ne0_prev) < tol
        converged_step0 = step;
        fprintf('  m=0 converged at step %d (z=%.1f um),  ne0=%.6f\n', ...
            step, step*Dz, ne0);
        break;
    end
    ne0_prev = ne0;
end
% Normalise fundamental mode to unit norm for projection
phi0_norm = phi0 / sqrt(sum(phi0.^2)*Dx);

%% =====================================================================
%% STEP 2: Compute the first higher-order mode (m=1) by deflation
%% =====================================================================
fprintf('\n=== Computing first higher-order mode (m=1) ===\n');

% Initial condition: antisymmetric (odd) function to favour m=1
%   Using x * Gaussian gives good overlap with the odd mode and
%   zero overlap with the even fundamental — speeds up convergence.
phi1 = x .* exp(-x.^2 / (2*(1.5*a)^2));

ne1_prev = 0;  ne1 = 0;
converged_step1 = n_steps;

for step = 1:n_steps

    % --- Refraction ---
    phi1 = prop_B .* phi1;

    % --- Diffraction ---
    PHI1 = fftshift(fft(ifftshift(phi1))) / N;
    PHI1 = prop_A .* PHI1;
    phi1 = real(fftshift(ifft(ifftshift(PHI1))) * N);

    % --- Orthogonalise against m=0 (Gram-Schmidt deflation) ---
    %   Remove any m=0 component that leaks in due to numerical errors
    overlap = sum(phi0_norm .* phi1) * Dx;
    phi1    = phi1 - overlap * phi0_norm;

    % --- Normalise ---
    phi1 = phi1 / max(abs(phi1));

    % --- Compute ne ---
    dphi1 = gradient(phi1, Dx);
    num1  = k^2*sum(Er.*phi1.^2)*Dx - sum(dphi1.^2)*Dx;
    den1  = k^2*sum(phi1.^2)*Dx;
    ne2   = real(num1/den1);
    if ne2 < 0
        % ne2 < 0 means this ne guess is below cutoff; clamp
        ne2 = n0^2 + 1e-10;
    end
    ne1   = sqrt(ne2);

    if step > 10 && abs(ne1 - ne1_prev) < tol
        converged_step1 = step;
        fprintf('  m=1 converged at step %d (z=%.1f um),  ne1=%.6f\n', ...
            step, step*Dz, ne1);
        break;
    end
    ne1_prev = ne1;
end

z_final = converged_step1 * Dz;

%% ===== Normalise m=1 mode for plotting =====
phi1_plot = phi1 / max(abs(phi1));

% Ensure positive lobe is on the left (cosmetic convention)
if phi1_plot(round(N/4)) < 0
    phi1_plot = -phi1_plot;
end

%% ===== Analytical TE / TM for the m=1 (ODD) mode =====
ne_TE1 = solve_dispersion_odd(k, n1, n0, a, 'TE');
ne_TM1 = solve_dispersion_odd(k, n1, n0, a, 'TM');

%% ===== Report =====
fprintf('\nResults:\n');
fprintf('  Numerical  ne (m=1) = %.4f\n', ne1);
fprintf('  Analytical TE ne    = %.4f\n', ne_TE1);
fprintf('  Analytical TM ne    = %.4f\n', ne_TM1);
fprintf('  Dx = %.4f um,  Dz = %.4f um,  z = %.4f um\n', Dx, Dz, z_final);

%% ===== Plot =====
figure('Color','w','Position',[100 100 950 520]);

plot(x, phi1_plot, 'b-', 'LineWidth', 2);
hold on;
yline(0, 'k--', 'LineWidth', 0.8);  % zero line to guide the eye
hold off;

xlim([-40  40]);
ylim([ -1    1]);
xlabel('x = n\Deltax (\mum)', 'FontSize', 13);
ylabel('\phi(n) (norm.)',      'FontSize', 13);

title(sprintf(['Numerical n_e = %.4f,  TE n_e = %.4f,  TM n_e = %.4f,  ' ...
               '\Deltax = %.4f (\\mum),  \Deltaz = %.4f (\\mum),  z = %.4f (\\mum)'], ...
    ne1, ne_TE1, ne_TM1, Dx, Dz, z_final), 'FontSize', 9);

grid on;
set(gca,'FontSize', 12);


%% =========================================================
function ne_out = solve_dispersion_odd(k, n1, n0, a, pol)
% Solve TE or TM dispersion relation for the first ODD guided mode.
%
% Odd-mode condition:
%   TE:  -kappa * cot(kappa*a) = gamma
%   TM:  -kappa * cot(kappa*a) = (n1/n0)^2 * gamma
%
% kappa = k*sqrt(n1^2 - ne^2),  gamma = k*sqrt(ne^2 - n0^2)
% Search ne in (n0, n1).

    ne_vec = linspace(n0 + 1e-9, n1 - 1e-9, 2e6);
    kappa  = k * sqrt(n1^2 - ne_vec.^2);
    gamma  = k * sqrt(ne_vec.^2 - n0^2);

    % Avoid poles of cot: wherever sin(kappa*a) ~ 0
    lhs = -kappa .* cos(kappa*a) ./ sin(kappa*a);   % -kappa*cot(kappa*a)

    if strcmpi(pol, 'TE')
        rhs = gamma;
    else   % TM
        rhs = (n1/n0)^2 * gamma;
    end

    f = lhs - rhs;

    % Mask out regions near poles of cot (sin ~ 0)
    pole_mask = abs(sin(kappa*a)) < 0.05;
    f(pole_mask) = NaN;

    % Find downward-going zero crossings (first one = m=1 odd mode)
    sgn = sign(f);
    cross_idx = find(diff(sgn) < 0 & ~isnan(sgn(1:end-1)) & ~isnan(sgn(2:end)), 1, 'first');

    if isempty(cross_idx)
        warning('No odd-mode root found for %s. Returning NaN.', pol);
        ne_out = NaN;
        return;
    end

    % Bisection refinement
    ne_lo = ne_vec(cross_idx);
    ne_hi = ne_vec(cross_idx + 1);

    for iter = 1:80
        ne_mid  = (ne_lo + ne_hi) / 2;
        kap_mid = k * sqrt(n1^2 - ne_mid^2);
        gam_mid = k * sqrt(ne_mid^2 - n0^2);
        if strcmpi(pol, 'TE')
            fmid = -kap_mid * cot(kap_mid*a) - gam_mid;
        else
            fmid = -kap_mid * cot(kap_mid*a) - (n1/n0)^2 * gam_mid;
        end

        kap_lo = k * sqrt(n1^2 - ne_lo^2);
        gam_lo = k * sqrt(ne_lo^2 - n0^2);
        if strcmpi(pol, 'TE')
            flo = -kap_lo * cot(kap_lo*a) - gam_lo;
        else
            flo = -kap_lo * cot(kap_lo*a) - (n1/n0)^2 * gam_lo;
        end

        if fmid * flo < 0
            ne_hi = ne_mid;
        else
            ne_lo = ne_mid;
        end
    end
    ne_out = (ne_lo + ne_hi) / 2;
end