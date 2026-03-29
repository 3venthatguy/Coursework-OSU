clear; clc; close all;

Kz = linspace(0, 2*pi, 1000);
Pa = cos(Kz).^2;
Pb = sin(Kz).^2;

figure;
plot(Kz, Pa, 'b', 'LineWidth', 1.5);
hold on;
plot(Kz, Pb, 'r', 'LineWidth', 1.5);

xlabel('Normalized distance \kappaz (unitless)');
ylabel('Normalized power (unitless)');
legend('|A(z)/A(0)|^2', '|B(z)/A(0)|^2');
xlim([0 2*pi]);
ylim([0 1]);
xticks([0 pi/2 pi 3*pi/2 2*pi]);
xticklabels({'0', '\pi/2', '\pi', '3\pi/2', '2\pi'});
grid on;