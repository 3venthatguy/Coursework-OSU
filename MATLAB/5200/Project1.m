clear; clc; close all;

rng_seed = 5200;
[x,M] = gen_input(rng_seed);

%% Task 1
fprintf('Task 1:\n')
L1 = 5;
L2 = 7;
freq = [0, 0.15, 0.25, 1];

h1 = firpm(L1-1, freq, [1 1 0 0]); % Lowpass filter h1[n]: length 5, order 4
h2 = firpm(L2-1, freq, [0 0 1 1]); % Highpass filter h2[n]: length 7, order 6

% Transpose from row vectors to column vectors
h1 = h1(:);
h2 = h2(:);
disp('h1:'); disp(h1);
disp('h2:'); disp(h2);

% Normalize true filters to same convention
h1_norm = h1/sum(abs(h1));
h1_norm = h1_norm/sign(h1_norm((L1+1)/2));
h2_norm = h2/sum(abs(h2));
h2_norm = h2_norm/sign(h2_norm((L2+1)/2));

y1 = conv(h1, x);
y2 = conv(h2, x);

Y1 = convmtx_lin(y1, L2);
Y2 = convmtx_lin(y2, L1);
A = [Y2, -Y1];

h = [h1; h2];
identity = A*h;

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
nerr_h = norm([h1_norm ; h2_norm] - [h1_est ; h2_est])/norm([h1_norm ; h2_norm]);
fprintf('Normalized error: %.2e\n', nerr_h);


%% Task 3
fprintf('\nTask 3:\n')
% Verification of conv_operator
h_test = randn(L2, 1);
z_test = randn(length(y1) + L2 - 1, 1);

% Forward check: Y1 * h should equal conv_operator(h, 'notransp', y1)
yu1 = conv_operator(h_test, 'notransp', y1);
fprintf('Forward error (Y1*h vs conv_operator): %.2e\n', norm(Y1 * h_test - yu1));

% Adjoint check: Y1.'*z should equal conv_operator(z, 'transp', y1)
yu1_star = conv_operator(z_test, 'transp', y1);
fprintf('Adjoint error (Y1^T*z vs conv_operator): %.2e\n', norm(Y1.' * z_test - yu1_star));

%% Task 5
fprintf('\nTask 5:\n')
gram_conv_handle = @(hin) gram_two_channel_conv(hin, L1, L2, y1, y2);
[~, lambda_max] = eigs(gram_conv_handle, L1+L2, 1, ...
    'largestabs', 'IsFunctionSymmetric', 1);

gram_conv_handle2 = @(hin) lambda_max*hin - gram_two_channel_conv(hin, L1, L2, y1, y2);
[v_eig, ~] = eigs(gram_conv_handle2, L1+L2, 1, ...
    'largestabs', 'IsFunctionSymmetric', 1);

v_task5 = v_eig(:, 1);
h1_est_task5 = v_task5(1:L1);
h2_est_task5 = v_task5(L1+(1:L2));

% Remove scaling ambiguity
h1_est_task5 = h1_est_task5/sum(abs(h1_est_task5));
h1_est_task5 = h1_est_task5/sign(h1_est_task5((L1+1)/2));
h2_est_task5 = h2_est_task5/sum(abs(h2_est_task5));
h2_est_task5 = h2_est_task5/sign(h2_est_task5((L2+1)/2));

% Normalized error
nerr_h_task5 = norm([h1_norm; h2_norm] - [h1_est_task5; h2_est_task5]) / norm([h1_norm; h2_norm]);
fprintf('Task 5 normalized error: %.2e\n', nerr_h_task5);


%% Functions
function [x,Lx] = gen_input(rng_seed)
    %Synthesizes a realistic (speech-like) input x via resonant filters (formants)
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
        % Note, x plays the role of input (y1, y2), h plays the role of the
        % filter (h1, h2)
        n = length(x);
        Y = convmtx_lin(h, n);
        y = Y * x;
    elseif strcmp(mode, 'transp')
        % Perform the adjoint operation: y = Y1^star * xin the 
        n = length(x) - length(h) + 1;
        Y = convmtx_lin(h, n);
        y = Y' * x;
    end
end

%% Task 4
function hout = gram_two_channel_conv(hin, L1, L2, y1, y2)
    % implement hout = A.’*A hin without constructing A
    h1 = hin(1:L1);
    h2 = hin(L1+(1:L2));

    % Compute inner term: w = Y2*h1 - Y1*h2
    w = conv_operator(h1, 'notransp', y2) - conv_operator(h2, 'notransp', y1);

    % Compute hout = A.'*w = [Y2^T; -Y1^T] * w
    hout = [conv_operator(w, 'transp', y2); -conv_operator(w, 'transp', y1)];
end