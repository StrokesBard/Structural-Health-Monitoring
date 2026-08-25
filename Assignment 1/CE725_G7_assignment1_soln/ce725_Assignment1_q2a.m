clc;
clear;
close all;

%% STRUCTURAL PARAMETERS (Table 1)

m1 = 10;   m2 = 10;
k1 = 10000; k2 = 10000;     % 10 kN/m → 10000 N/m
c1 = 31.76; c2 = 31.76;

M = [m1 0;
     0  m2];

K = [k1+k2  -k2;
     -k2     k2];

C = [c1+c2  -c2;
     -c2     c2];

influence = [1;1];

%% CONTINUOUS-TIME STATE-SPACE MODEL
% State: X = [y1 y2 y1_dot y2_dot]'

Ac = [ zeros(2)   eye(2);
      -M\K       -M\C ];

Bc = [ zeros(2,1);
      -influence ];

%% LOAD DATA

% Ground motion (50 Hz)
gm = load('HW1_Q2_El_Centro.txt');
t_gm = gm(:,1);
ug_ddot_raw = gm(:,2);

% Measurements (200 Hz)
meas = load('HW1_Q2_Measurements.txt');
t = meas(:,1);
y_meas = meas(:,2:3);      % displacement
a_meas = meas(:,4:5);      % absolute acceleration

dt = t(2) - t(1);
N = length(t);

% Interpolate ground motion to 200 Hz
ug_ddot = interp1(t_gm, ug_ddot_raw, t, 'linear','extrap');

%% DISCRETIZE SYSTEM (Exact Discretization)

Phi   = expm(Ac*dt);
Gamma = (Phi - eye(4))*(Ac\Bc);

%% MEASUREMENT MODEL (ACCELERATION ONLY)
% Absolute acceleration:
% y_ddot + ug_ddot = -M^-1 K y - M^-1 C y_dot

H = [-M\K   -M\C];

% Noise variances (Table 2)
var_a = [0.105  0.162];
R = diag(var_a);

Z_all = a_meas';

%% INITIAL CONDITIONS

X0 = zeros(4,1);        % structure initially at rest
P0 = eye(4)*1e-4;

%% RUN KALMAN FILTER – CASE 1: Q = 0

Q_zero = zeros(4);
X = X0;
P = P0;
X_est_zero = zeros(4,N);

for k = 1:N
    
    % Prediction
    X_pred = Phi*X + Gamma*ug_ddot(k);
    P_pred = Phi*P*Phi' + Q_zero;
    
    % Update
    Z = Z_all(:,k);
    K_gain = P_pred*H'/(H*P_pred*H' + R);
    
    X = X_pred + K_gain*(Z - H*X_pred);
    P = (eye(4) - K_gain*H)*P_pred;
    
    X_est_zero(:,k) = X;
end

%% RUN KALMAN FILTER – CASE 2: Q ≠ 0

Q_small = eye(4)*1e-7;   % small process noise
X = X0;
P = P0;
X_est_smallQ = zeros(4,N);

for k = 1:N
    
    % Prediction
    X_pred = Phi*X + Gamma*ug_ddot(k);
    P_pred = Phi*P*Phi' + Q_small;
    
    % Update
    Z = Z_all(:,k);
    K_gain = P_pred*H'/(H*P_pred*H' + R);
    
    X = X_pred + K_gain*(Z - H*X_pred);
    P = (eye(4) - K_gain*H)*P_pred;
    
    X_est_smallQ(:,k) = X;
end

%% PLOT RESULTS – DISPLACEMENTS

figure;
subplot(2,1,1)
plot(t, X_est_zero(1,:), 'b','LineWidth',1.2); hold on;
plot(t, X_est_smallQ(1,:), 'r--','LineWidth',1.2);
title('Story 1 Displacement (Acceleration Only)');
ylabel('y_1 (m)');
legend('Q = 0','Q ≠ 0');
grid on

subplot(2,1,2)
plot(t, X_est_zero(2,:), 'b','LineWidth',1.2); hold on;
plot(t, X_est_smallQ(2,:), 'r--','LineWidth',1.2);
ylabel('y_2 (m)');
xlabel('Time (s)');
legend('Q = 0','Q ≠ 0');
grid on

%% VELOCITY RESPONSE

figure;
subplot(2,1,1)
plot(t, X_est_zero(3,:), 'b','LineWidth',1.2); hold on;
plot(t, X_est_smallQ(3,:), 'r--','LineWidth',1.2);
title('Story 1 Velocity');
ylabel('y_1 dot (m/s)');
legend('Q = 0','Q ≠ 0');
grid on

subplot(2,1,2)
plot(t, X_est_zero(4,:), 'b','LineWidth',1.2); hold on;
plot(t, X_est_smallQ(4,:), 'r--','LineWidth',1.2);
ylabel('y_2 dot (m/s)');
xlabel('Time (s)');
legend('Q = 0','Q ≠ 0');
grid on