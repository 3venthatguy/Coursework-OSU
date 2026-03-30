clear; clc; close all;

rng_seed = 5200;
[x,M] = gen_input(rng_seed);

L1 = 5;
L2 = 7;
freq = [0, 0.15, 0.25, 1];

h1 = firpm(L1, freq, [1 1 0 0]); % Lowpass filter h1[n]: length 5, order 4
h2 = firpm(L2, freq, [0 0 1 1]); % Highpass filter h2[n]: length 7, order 6

y1 = conv(x, h1);
y2 = conv(x, h2);

Y1 = convmtx_lin(y1, L2);
Y2 = convmtx_lin(y2, L1);
A = [Y2, -Y1];

[~, lambda_max] = eigs(A'*A, 1, 'largestabs');
[v, ~] = eigs(lambda_max*eye(L1+L2) - A'*A, 1, 'largestabs');

% Extract estimated filter vectors
h1_est = v(1:L1);
h2_est = v(L1+(1:L2));

% Remove scaling ambiguity
h1_est = h1_est/sum(abs(h1_est));
h1_est = h1_est/sign(h1_est((L1+1)/2));
h2_est = h2_est/sum(abs(h2_est));
h2_est = h2_est/sign(h2_est((L2+1)/2));

% Verify normalized error less than 10^-9
nerr_h = norm([h1 ; h2] - [h1_est ; h2_est])/norm([h1 ; h2]);

function [x,Lx] = gen_input(rng_seed)
    % Synthesizes a realistic (speech-like) input x via resonant filters (formants)
    rng(rng_seed);
    
    %% ---------------- Parameters ----------------
    Fs   = 8000;              % "Audio"-like sampling rate (Hz), used for input synthesis
    T    = 0.25;              % Duration (s) for the input
    Lx   = round(Fs*T);       % Input length
    
    %% ---------------- Realistic input synthesis (speech-like) ----------------
    % Create voiced-speech-like signal by exciting two formant resonances.
    % Steps:
    %  1) Start with white Gaussian noise as a broadband excitation.
    %  2) Shape it through two second-order resonant IIR filters (formants).
    %  3) Apply a gentle amplitude envelope (attack/decay) to mimic articulation.
    
    exc = randn(Lx,1);                   % broadband excitation
    
    % Two formants: around F1 ~ 700 Hz, F2 ~ 1200 Hz (typical for a vowel), moderate Q
    F1 = 700;  BW1 = 100;
    F2 = 1200; BW2 = 150;
    
    [b1, a1] = biquad_formant(F1, BW1, Fs);
    [b2, a2] = biquad_formant(F2, BW2, Fs);
    
    x_formants = filter(b1, a1, exc);
    x_formants = filter(b2, a2, x_formants);
    
    % Smooth amplitude envelope (fade in/out)
    n = (0:Lx-1)'; 
    env = 0.5*(1 - cos(2*pi*min(n, Lx-1)/ (Lx-1)));  % raised-cosine
    x = x_formants .* env;
    
    % Normalize input power
    x = x / max(max(abs(x)), 1e-6);
end

function [b,a] = biquad_formant(Fc, BW, Fs)
    % Design a simple second-order resonator (biquad) with center frequency Fc and
    % 3-dB bandwidth BW at sampling rate Fs. Produces a "formant-like" peak.
    % Based on a normalized digital resonator: H(z) = (1 - r) / (1 - 2r cos(w0) z^-1 + r^2 z^-2)
    % where r controls bandwidth: BW ≈ (1 - r) * Fs / pi for small BW.
    w0 = 2*pi*Fc/Fs;
    r  = max(0.0, min(0.999, 1 - (BW*pi)/Fs));  % clamp r for stability
    b  = [(1 - r), 0, 0];
    a  = [1, -2*r*cos(w0), r^2];
end

function Y = convmtx_lin(x, n)
    x = x(:); m = length(x); L = m + n - 1;
    Y = zeros(L, n);
    for k = 1:n
        Y(k:k+m-1, k) = x;
    end
end

function y = conv_operator(x, mode, h)
    if strcmp(mode, 'notransp')
        % Perform the forward operation: y = Y1 * x
        elseif strcmp(mode, 'transp')
        % Perform the adjoint operation: y = Y1^star * x
    end
end
