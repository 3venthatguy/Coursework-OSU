n  = -5:30;
u  = (n >= 0);

x1 = cos(3*n) .* u;
x2 = 3 .* (0.8.^n) .* cos(pi*n/4) .* u;
x3 = (-0.5).^n .* u;
x4 = cos(pi*n/5 - pi/2);
x5 = cos(n);

figure; tiledlayout(3,2);

nexttile; stem(n,x1,'filled'); title('x_1[n]'); grid on
nexttile; stem(n,x2,'filled'); title('x_2[n]'); grid on
nexttile; stem(n,x3,'filled'); title('x_3[n]'); grid on
nexttile; stem(n,x4,'filled'); title('x_4[n]'); grid on
nexttile; stem(n,x5,'filled'); title('x_5[n]'); grid on