% File gen_input

function [x,Lx] = gen_input(rng_seed)
% Synthesizes a realistic (speech-like) input x via resonant filters (formants)
rng(rng_seed);

%% ---------------- Parameters ----------------
Fs   = 8000;              % "Audio"-like sampling rate (Hz), used for input synthesis
T    = 0.25;              % Duration (s) for the input
Lx   = round(Fs*T);       % Input length
Lh   = 8;                 % FIR channel length (assumed known)

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
