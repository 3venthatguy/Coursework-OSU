clear; clc; close all;

V=380/sqrt(3);
R1=0.1; R2=0.12; Xeq=0.75;
p=6; f=60; n_s=120*f/p;
omega_s=n_s*pi/30;
P_friction=100;

n_r=0:1:n_s;
s=(n_s-n_r)/n_s;
Z1=(R1+R2./s).^2 + Xeq*Xeq;
Td=3*V*V*R2./s/omega_s./Z1;

I = V./sqrt(Z1);
P_ag = 3.*I.*I.*R2./s;     % Air gap
P_in = 3*V^2*(R1 + R2./s)./Z1;
P_out = P_ag.*(1 - s) - P_friction;

subplot(311)
plot(n_r, Td, 'LineWidth', 2.5, 'Color', [0, 0.4470, 0.7410]);
grid on;
xlabel('Rotor Mechanical Speed (rpm)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Torque (N·m)', 'FontSize', 14, 'FontWeight', 'bold');
title('Torque vs Rotor Mechanical Speed', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
xlim([0 n_s]);
ylim([0 max(Td)*1.1]);

subplot(312)
plot(n_r, P_in/1000, 'LineWidth', 2.5, 'Color', [0.8500, 0.3250, 0.0980], 'DisplayName', 'Input Power');
hold on;
plot(n_r, P_out/1000, 'LineWidth', 2.5, 'Color', [0, 0.5, 0], 'DisplayName', 'Output Power');
grid on;
xlabel('Rotor Mechanical Speed (rpm)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Power (kW)', 'FontSize', 14, 'FontWeight', 'bold');
title('Input and Output Power vs Rotor Mechanical Speed', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 12);
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
xlim([0 n_s]);
ylim([0 max(P_in/1000)*1.1]);

subplot(313)
plot(n_r, I, 'LineWidth', 2.5, 'Color', [0.4940, 0.1840, 0.5560]);
grid on;
xlabel('Rotor Mechanical Speed (rpm)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Stator Current (A)', 'FontSize', 14, 'FontWeight', 'bold');
title('Stator Current vs Rotor Mechanical Speed', 'FontSize', 16, 'FontWeight', 'bold');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
xlim([0 n_s]);
ylim([0 max(I)*1.1]);