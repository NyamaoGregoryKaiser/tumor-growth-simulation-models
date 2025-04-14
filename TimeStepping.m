%% Adaptive Time Stepping for Tumor Growth Models
% This script implements an adaptive time stepping algorithm for a tumor growth model
% and compares different time-stepping strategies for efficiency and accuracy.
% Features enhanced visualization with better colormaps, consistent formatting, and animated output.

clear all; close all; clc;

%% Parameters for tumor growth model
% Model parameters
D = 0.5;          % Diffusion coefficient for tumor cells
r = 0.2;          % Growth rate
K = 1.0;          % Carrying capacity
alpha = 0.1;      % Death rate due to therapy

% Domain parameters
L = 10;           % Domain size
N = 100;          % Number of spatial points
dx = L/N;         % Spatial step size
x = linspace(0, L, N);
y = linspace(0, L, N);
[X, Y] = meshgrid(x, y);

% Initial condition: Gaussian tumor distribution
sigma = 0.5;
u0 = exp(-((X-L/2).^2 + (Y-L/2).^2)/(2*sigma^2));

% Time simulation parameters
Tfinal = 20;       % Final time
plotSteps = 10;    % Number of plots to make
saveAnimation = true; % Whether to save animations

% Set up color schemes and visualization parameters
viridisColors = viridisColormap();
defaultFontSize = 12;
lineWidth = 2;
markerSize = 8;

% Create output directory for animations if needed
if saveAnimation && ~exist('animations', 'dir')
    mkdir('animations');
end

%% Run simulations with different time stepping methods
methods = {'Fixed', 'Adaptive', 'RK45'};
methodColors = {[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.4660, 0.6740, 0.1880]};
errors = zeros(length(methods), 1);
cputimes = zeros(length(methods), 1);
time_history = cell(length(methods), 1);
solution_history = cell(length(methods), 1);

% Set up figure properties
set(0, 'DefaultAxesFontSize', defaultFontSize);
set(0, 'DefaultTextFontSize', defaultFontSize);
set(0, 'DefaultLineLineWidth', lineWidth);

% Reference solution with very small time step
fprintf('Computing reference solution...\n');
dt_ref = 0.001;
[u_ref, ~, ~, time_points_ref, solution_snapshots_ref] = simulateTumorGrowth(u0, D, r, K, alpha, dx, dt_ref, Tfinal, 'Fixed', false, plotSteps);

% Run each method
for m = 1:length(methods)
    method = methods{m};
    fprintf('Running %s time stepping method...\n', method);
    
    % Set initial time step based on method
    if strcmp(method, 'Fixed')
        dt_init = 0.01;
    else
        dt_init = 0.05;
    end
    
    % Run simulation and time it
    tic;
    [u_final, time_steps, errors_local, time_points, solution_snapshots] = simulateTumorGrowth(u0, D, r, K, alpha, dx, dt_init, Tfinal, method, true, plotSteps);
    cputimes(m) = toc;
    
    % Store time evolution data
    time_history{m} = time_points;
    solution_history{m} = solution_snapshots;
    
    % Calculate error compared to reference solution
    errors(m) = norm(u_final(:) - u_ref(:)) / norm(u_ref(:));
    
    % Create enhanced figures
    createDetailedPlots(X, Y, u_final, u_ref, time_steps, errors_local, x, method, methodColors{m});
    
    % Create animation
    if saveAnimation
        createAnimation(X, Y, x, solution_snapshots, time_points, method, methodColors{m});
    end
end

% Create summary comparison plots
createComparisonPlots(methods, errors, cputimes, methodColors);

% Create evolution comparison plot
createEvolutionComparisonPlot(X, Y, x, methods, time_history, solution_history, methodColors);

%% Function to run tumor growth simulation with various time stepping methods
function [u, time_steps, error_estimates, time_points, solution_snapshots] = simulateTumorGrowth(u0, D, r, K, alpha, dx, dt_init, Tfinal, method, verbose, numSnapshots)
    % Initialize
    u = u0;
    t = 0;
    dt = dt_init;
    time_steps = [0, dt];  % Store time steps
    error_estimates = [];  % For adaptive methods
    
    % Initialize solution history for visualization
    snapshot_times = linspace(0, Tfinal, numSnapshots);
    next_snapshot_idx = 1;
    time_points = [0];
    solution_snapshots = {u0};
    
    % Parameters for adaptive time stepping
    tol = 1e-3;      % Error tolerance
    dt_min = 1e-4;   % Minimum time step
    dt_max = 0.1;    % Maximum time step
    safety = 0.8;    % Safety factor
    
    % Main time loop
    while t < Tfinal
        % Adjust time step to hit Tfinal exactly
        if t + dt > Tfinal
            dt = Tfinal - t;
        end
        
        % Apply time stepping method
        switch method
            case 'Fixed'
                % Forward Euler with fixed time step
                u_new = stepTumor(u, D, r, K, alpha, dx, dt);
                
            case 'Adaptive'
                % Adaptive time stepping based on local error estimate
                % Try a step with dt and dt/2
                u_full = stepTumor(u, D, r, K, alpha, dx, dt);
                
                % Two half steps
                u_half = stepTumor(u, D, r, K, alpha, dx, dt/2);
                u_half = stepTumor(u_half, D, r, K, alpha, dx, dt/2);
                
                % Estimate local error
                local_error = norm(u_full(:) - u_half(:)) / norm(u_half(:));
                error_estimates = [error_estimates; local_error];
                
                % Adjust time step based on error
                if local_error > tol
                    % Reject step and reduce dt
                    dt = max(safety * dt * (tol/local_error)^0.5, dt_min);
                    continue;  % Retry with smaller step
                else
                    % Accept step and adjust dt for next step
                    dt_new = min(safety * dt * (tol/local_error)^0.5, dt_max);
                    u_new = u_half;  % Use more accurate solution
                    
                    % Only change dt if significantly different
                    if dt_new > 1.2*dt || dt_new < 0.8*dt
                        dt = dt_new;
                    end
                end
                
            case 'RK45'
                % Runge-Kutta 4-5 method (Dormand-Prince)
                % Butcher tableau for Dormand-Prince method
                a = [0, 0, 0, 0, 0, 0;
                     1/5, 0, 0, 0, 0, 0;
                     3/40, 9/40, 0, 0, 0, 0;
                     44/45, -56/15, 32/9, 0, 0, 0;
                     19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0;
                     9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0];
                b4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84];
                b5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40];
                
                % Calculate k values
                k = zeros(size(u, 1), size(u, 2), 7);
                k(:,:,1) = tumorRHS(u, D, r, K, alpha, dx);
                
                u_temp = u + dt * (a(2,1) * k(:,:,1));
                k(:,:,2) = tumorRHS(u_temp, D, r, K, alpha, dx);
                
                u_temp = u + dt * (a(3,1) * k(:,:,1) + a(3,2) * k(:,:,2));
                k(:,:,3) = tumorRHS(u_temp, D, r, K, alpha, dx);
                
                u_temp = u + dt * (a(4,1) * k(:,:,1) + a(4,2) * k(:,:,2) + a(4,3) * k(:,:,3));
                k(:,:,4) = tumorRHS(u_temp, D, r, K, alpha, dx);
                
                u_temp = u + dt * (a(5,1) * k(:,:,1) + a(5,2) * k(:,:,2) + a(5,3) * k(:,:,3) + a(5,4) * k(:,:,4));
                k(:,:,5) = tumorRHS(u_temp, D, r, K, alpha, dx);
                
                u_temp = u + dt * (a(6,1) * k(:,:,1) + a(6,2) * k(:,:,2) + a(6,3) * k(:,:,3) + a(6,4) * k(:,:,4) + a(6,5) * k(:,:,5));
                k(:,:,6) = tumorRHS(u_temp, D, r, K, alpha, dx);
                
                % Fourth-order solution
                u4 = u;
                for i = 1:6
                    u4 = u4 + dt * b4(i) * k(:,:,i);
                end
                
                % Fifth-order solution for error estimation
                u5 = u;
                for i = 1:6
                    u5 = u5 + dt * b5(i) * k(:,:,i);
                end
                k(:,:,7) = tumorRHS(u5, D, r, K, alpha, dx);
                u5 = u5 + dt * b5(7) * k(:,:,7);
                
                % Error estimation
                local_error = norm(u5(:) - u4(:)) / norm(u5(:));
                error_estimates = [error_estimates; local_error];
                
                % Adapt time step
                if local_error > tol
                    % Reject step and reduce dt
                    dt = max(safety * dt * (tol/local_error)^0.2, dt_min);
                    continue;  % Retry with smaller step
                else
                    % Accept step and adjust dt for next step
                    dt_new = min(safety * dt * (tol/local_error)^0.2, dt_max);
                    u_new = u5;  % Use fifth-order solution
                    
                    % Only change dt if significantly different
                    if dt_new > 1.2*dt || dt_new < 0.8*dt
                        dt = dt_new;
                    end
                end
        end
        
        % Update solution and time
        u = u_new;
        t = t + time_steps(end);
        time_steps = [time_steps, dt];
        
        % Store solution snapshots at specified times
        while next_snapshot_idx <= length(snapshot_times) && t >= snapshot_times(next_snapshot_idx)
            time_points = [time_points, t];
            solution_snapshots{end+1} = u;
            next_snapshot_idx = next_snapshot_idx + 1;
        end
        
        % Display progress
        if verbose && mod(length(time_steps), 10) == 0
            fprintf('t = %.3f (%.1f%%), dt = %.5f\n', t, 100*t/Tfinal, dt);
        end
    end
    
    % Ensure we have the final solution
    if time_points(end) < Tfinal
        time_points = [time_points, Tfinal];
        solution_snapshots{end+1} = u;
    end
end

%% Function for a single time step using forward Euler
function u_new = stepTumor(u, D, r, K, alpha, dx, dt)
    % Apply tumor growth model using forward Euler in time,
    % central difference in space for diffusion
    
    % Calculate Laplacian using central difference
    u_lap = zeros(size(u));
    
    % Interior points
    for i = 2:size(u,1)-1
        for j = 2:size(u,2)-1
            u_lap(i,j) = (u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4*u(i,j)) / dx^2;
        end
    end
    
    % Neumann boundary conditions (zero flux)
    % Top and bottom boundaries
    u_lap(1,:) = (2*u(2,:) - 2*u(1,:)) / dx^2;
    u_lap(end,:) = (2*u(end-1,:) - 2*u(end,:)) / dx^2;
    
    % Left and right boundaries
    u_lap(:,1) = (2*u(:,2) - 2*u(:,1)) / dx^2;
    u_lap(:,end) = (2*u(:,end-1) - 2*u(:,end)) / dx^2;
    
    % Logistic growth with therapy term
    reaction = r * u .* (1 - u/K) - alpha * u;
    
    % Update solution using forward Euler
    u_new = u + dt * (D * u_lap + reaction);
    
    % Ensure non-negativity
    u_new = max(u_new, 0);
end

%% Function to calculate right-hand side of tumor growth PDE
function dudt = tumorRHS(u, D, r, K, alpha, dx)
    % Calculate Laplacian using central difference
    u_lap = zeros(size(u));
    
    % Interior points
    for i = 2:size(u,1)-1
        for j = 2:size(u,2)-1
            u_lap(i,j) = (u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4*u(i,j)) / dx^2;
        end
    end
    
    % Neumann boundary conditions (zero flux)
    % Top and bottom boundaries
    u_lap(1,:) = (2*u(2,:) - 2*u(1,:)) / dx^2;
    u_lap(end,:) = (2*u(end-1,:) - 2*u(end,:)) / dx^2;
    
    % Left and right boundaries
    u_lap(:,1) = (2*u(:,2) - 2*u(:,1)) / dx^2;
    u_lap(:,end) = (2*u(:,end-1) - 2*u(:,end)) / dx^2;
    
    % Logistic growth with therapy term
    reaction = r * u .* (1 - u/K) - alpha * u;
    
    % Right-hand side of PDE
    dudt = D * u_lap + reaction;
end

%% Function to create the Viridis colormap (perceptually uniform)
function cm = viridisColormap()
    % A perceptually uniform colormap from matplotlib
    cm = [
        0.267004, 0.004874, 0.329415;
        0.275191, 0.032909, 0.390531;
        0.280935, 0.059733, 0.444230;
        0.283912, 0.086992, 0.490252;
        0.284036, 0.114715, 0.527551;
        0.281197, 0.143891, 0.555343;
        0.275713, 0.174556, 0.573456;
        0.267986, 0.206336, 0.580984;
        0.258505, 0.238613, 0.577399;
        0.247997, 0.271080, 0.563065;
        0.237127, 0.303476, 0.538456;
        0.226106, 0.335546, 0.504365;
        0.215047, 0.367152, 0.460551;
        0.204262, 0.398186, 0.407994;
        0.194107, 0.428455, 0.348444;
        0.184935, 0.458005, 0.283787;
        0.177423, 0.486947, 0.216215;
        0.172234, 0.515401, 0.149661;
        0.169456, 0.543552, 0.087953;
        0.169326, 0.571483, 0.035620;
        0.172940, 0.599191, 0.001261;
        0.181706, 0.626696, 0.000000;
        0.195719, 0.653920, 0.000000;
        0.214847, 0.680689, 0.000000;
        0.238547, 0.706769, 0.000000;
        0.266079, 0.731940, 0.000000;
        0.297456, 0.756002, 0.000000;
        0.332295, 0.778718, 0.000000;
        0.370207, 0.799892, 0.000000;
        0.410571, 0.819387, 0.000000;
        0.452675, 0.837097, 0.000000;
        0.495976, 0.853040, 0.000000;
        0.539897, 0.867301, 0.000000;
        0.584638, 0.880046, 0.000000;
        0.629948, 0.891507, 0.000000;
        0.675303, 0.901935, 0.000000;
        0.720088, 0.911536, 0.000000;
        0.763906, 0.920637, 0.000000;
        0.806327, 0.929613, 0.000000;
        0.846844, 0.938900, 0.000000;
        0.885007, 0.948905, 0.000000;
        0.920419, 0.959916, 0.000000;
        0.952745, 0.972080, 0.000000;
        0.981683, 0.985493, 0.000000;
        0.993841, 1.000000, 0.001174
    ];
end

%% Function to create nice turbo colormap (alternative to jet)
function cm = turboColormap()
    % A more perceptually uniform, colorblind-friendly alternative to jet
    cm = [
        0.18995,0.07176,0.23217;
        0.21859,0.08831,0.30046;
        0.24701,0.10549,0.36877;
        0.27427,0.12348,0.43404;
        0.30021,0.14250,0.49306;
        0.32440,0.16262,0.54306;
        0.34646,0.18391,0.58201;
        0.36617,0.20627,0.60898;
        0.38335,0.22963,0.62450;
        0.39805,0.25388,0.63036;
        0.41044,0.27882,0.62931;
        0.42068,0.30428,0.62453;
        0.42899,0.33005,0.61798;
        0.43559,0.35597,0.61104;
        0.44062,0.38185,0.60494;
        0.44427,0.40760,0.60031;
        0.44672,0.43310,0.59756;
        0.44807,0.45828,0.59700;
        0.44844,0.48307,0.59880;
        0.44791,0.50740,0.60299;
        0.44653,0.53121,0.60957;
        0.44437,0.55445,0.61839;
        0.44150,0.57709,0.62934;
        0.43799,0.59912,0.64220;
        0.43389,0.62048,0.65676;
        0.42924,0.64119,0.67283;
        0.42407,0.66122,0.69020;
        0.41845,0.68056,0.70871;
        0.41241,0.69923,0.72812;
        0.40600,0.71722,0.74832;
        0.39926,0.73453,0.76913;
        0.39226,0.75116,0.79043;
        0.38503,0.76712,0.81214;
        0.37760,0.78240,0.83414;
        0.37003,0.79702,0.85632;
        0.36237,0.81100,0.87859;
        0.35468,0.82436,0.90087;
        0.34703,0.83713,0.92307;
        0.33946,0.84937,0.94516;
        0.33203,0.86108,0.96707;
        0.32479,0.87234,0.98878;
        0.31777,0.88322,1.01014;
        0.31107,0.89376,1.03111;
        0.30474,0.90398,1.05165;
        0.29881,0.91389,1.07170;
        0.29332,0.92354,1.09125;
        0.28830,0.93295,1.11027;
        0.28375,0.94214,1.12874;
        0.27969,0.95114,1.14664;
        0.27612,0.95996,1.16396;
        0.27304,0.96861,1.18070;
        0.27045,0.97710,1.19687;
        0.26835,0.98546,1.21247;
        0.26674,0.99368,1.22752;
        0.26562,1.00176,1.24205;
        0.26500,1.00972,1.25605;
        0.26486,1.01755,1.26954;
        0.26519,1.02529,1.28255;
        0.26599,1.03293,1.29511;
        0.26724,1.04048,1.30722;
        0.26895,1.04796,1.31893;
        0.27111,1.05537,1.33024;
        0.27367,1.06273,1.34118;
        0.27661,1.07003,1.35178;
        0.27992,1.07730,1.36207;
        0.28358,1.08453,1.37206;
        0.28753,1.09173,1.38178;
        0.29176,1.09892,1.39126;
        0.29624,1.10608,1.40052;
        0.30095,1.11323,1.40958;
        0.30587,1.12037,1.41843;
        0.31099,1.12750,1.42710;
        0.31628,1.13462,1.43558;
        0.32172,1.14175,1.44391;
        0.32731,1.14887,1.45206;
        0.33301,1.15598,1.46005;
        0.33883,1.16310,1.46786;
        0.34474,1.17021,1.47548;
        0.35074,1.17733,1.48290;
        0.35682,1.18444,1.49010;
        0.36297,1.19156,1.49707;
        0.36918,1.19867,1.50380;
        0.37544,1.20580,1.51028;
        0.38175,1.21293,1.51649;
        0.38809,1.22004,1.52244;
        0.39445,1.22717,1.52809;
        0.40085,1.23429,1.53345;
        0.40725,1.24142,1.53849;
        0.41368,1.24855,1.54321;
        0.42012,1.25567,1.54760;
        0.42656,1.26279,1.55166;
        0.43300,1.26990,1.55538;
        0.43944,1.27701,1.55874;
        0.44587,1.28411,1.56175;
        0.45229,1.29120,1.56439;
        0.45869,1.29828,1.56666;
        0.46508,1.30535,1.56855;
        0.47144,1.31242,1.57006;
        0.47777,1.31947,1.57118;
        0.48408,1.32651,1.57192;
        0.49038,1.33354,1.57225;
        0.49665,1.34055,1.57219;
        0.50289,1.34755,1.57173;
        0.50912,1.35453,1.57086;
        0.51532,1.36149,1.56961;
        0.52149,1.36844,1.56794;
        0.52765,1.37538,1.56587;
        0.53380,1.38229,1.56341;
        0.53992,1.38918,1.56054;
        0.54605,1.39606,1.55728;
        0.55218,1.40293,1.55362;
        0.55831,1.40977,1.54957;
        0.56443,1.41659,1.54512;
        0.57055,1.42339,1.54031;
        0.57667,1.43019,1.53510;
        0.58279,1.43697,1.52952;
        0.58891,1.44373,1.52357;
        0.59502,1.45047,1.51725;
        0.60114,1.45720,1.51056;
        0.60724,1.46393,1.50352;
        0.61334,1.47063,1.49613;
        0.61943,1.47732,1.48839;
        0.62552,1.48399,1.48031;
        0.63159,1.49066,1.47189;
        0.63766,1.49730,1.46315;
        0.64371,1.50394,1.45409;
        0.64975,1.51056,1.44472;
        0.65577,1.51717,1.43505;
        0.66178,1.52377,1.42507;
        0.66777,1.53036,1.41481;
        0.67374,1.53693,1.40427;
        0.67969,1.54349,1.39344;
        0.68563,1.55004,1.38235;
        0.69154,1.55658,1.37098;
        0.69744,1.56310,1.35936;
        0.70331,1.56961,1.34748;
        0.70916,1.57611,1.33536;
        0.71498,1.58260,1.32301;
        0.72077,1.58908,1.31041;
        0.72654,1.59554,1.29760;
        0.73229,1.60199,1.28456;
        0.73801,1.60842,1.27131;
        0.74369,1.61485,1.25785;
        0.74934,1.62126,1.24418;
        0.75495,1.62766,1.23034;
        0.76053,1.63404,1.21630;
        0.76608,1.64041,1.20209;
        0.77159,1.64676,1.18771;
        0.77706,1.65310,1.17316;
        0.78249,1.65942,1.15848;
        0.78787,1.66573,1.14365;
        0.79322,1.67202,1.12868;
        0.79852,1.67829,1.11359;
        0.80378,1.68455,1.09841;
        0.80899,1.69079,1.08312;
        0.81416,1.69701,1.06774;
        0.81928,1.70321,1.05228;
        0.82435,1.70940,1.03674;
        0.82938,1.71556,1.02114;
        0.83435,1.72171,1.00549;
        0.83928,1.72783,0.98979;
        0.84416,1.73394,0.97407;
        0.84898,1.74003,0.95832;
        0.85375,1.74609,0.94256;
        0.85847,1.75214,0.92682;
        0.86314,1.75817,0.91108;
        0.86775,1.76418,0.89538;
        0.87231,1.77016,0.87973;
        0.87682,1.77613,0.86413;
        0.88127,1.78207,0.84860;
        0.88566,1.78799,0.83316;
        0.89001,1.79389,0.81780;
        0.89429,1.79977,0.80255;
        0.89853,1.80563,0.78740;
        0.90270,1.81146,0.77237;
        0.90683,1.81727,0.75747;
        0.91090,1.82306,0.74271;
        0.91492,1.82883,0.72810;
        0.91888,1.83458,0.71364;
        0.92280,1.84031,0.69933;
        0.92665,1.84601,0.68520;
        0.93046,1.85169,0.67123;
        0.93423,1.85736,0.65745;
        0.93794,1.86299,0.64385;
        0.94160,1.86861,0.63045;
        0.94521,1.87420,0.61725;
        0.94876,1.87977,0.60428;
        0.95226,1.88533,0.59151;
        0.95570,1.89087,0.57897;
        0.95909,1.89638,0.56665;
        0.96242,1.90187,0.55458;
        0.96570,1.90735,0.54276;
        0.96892,1.91281,0.53120;
        0.97208,1.91825,0.51991;
        0.97519,1.92367,0.50891;
        0.97823,1.92908,0.49820;
        0.98122,1.93447,0.48780;
        0.98415,1.93984,0.47770;
        0.98702,1.94520,0.46792;
        0.98982,1.95053,0.45845;
        0.99257,1.95586,0.44931;
        0.99525,1.96116,0.44050;
        0.99786,1.96645,0.43203;
        1.00042,1.97173,0.42390;
        1.00290,1.97698,0.41613;
        1.00531,1.98223,0.40871;
        1.00764,1.98745,0.40166;
        1.00989,1.99266,0.39499;
        1.01207,1.99785,0.38868;
        1.01416,2.00303,0.38275;
        1.01617,2.00818,0.37721;
        1.01810,2.01332,0.37205;
        1.01995,2.01843,0.36729;
        1.02170,2.02354,0.36294;
        1.02336,2.02862,0.35900;
        1.02494,2.03369,0.35549;
        1.02642,2.03874,0.35242;
        1.02780,2.04377,0.34981;
        1.02909,2.04878,0.34765;
        1.03026,2.05377,0.34597;
        1.03133,2.05874,0.34479;
        1.03230,2.06370,0.34409;
        1.03316,2.06863,0.34391;
        1.03393,2.07355,0.34426;
        1.03459,2.07846,0.34511;
        1.03515,2.08334,0.34646;
        1.03560,2.08820,0.34830;
        1.03596,2.09305,0.35066;
        1.03622,2.09788,0.35353;
        1.03638,2.10268,0.35689;
        1.03646,2.10748,0.36076;
        1.03644,2.11224,0.36513;
        1.03634,2.11699,0.37002;
        1.03616,2.12172,0.37540;
        1.03589,2.12643,0.38130;
        1.03555,2.13113,0.38771;
        1.03513,2.13580,0.39464;
        1.03464,2.14045,0.40209;
        1.03408,2.14509,0.41009;
        1.03347,2.14971,0.41862;
        1.03279,2.15431,0.42769;
        1.03206,2.15890,0.43730;
        1.03127,2.16346,0.44746;
        1.03044,2.16801,0.45814;
        1.02957,2.17253,0.46938;
        1.02867,2.17704,0.48112;
        1.02773,2.18153,0.49342;
        1.02676,2.18601,0.50628;
        1.02577,2.19047,0.51966;
        1.02476,2.19490,0.53358;
        1.02375,2.19933,0.54802;
        1.02272,2.20373,0.56297;
        1.02170,2.20812,0.57843;
        1.02068,2.21248,0.59439;
        1.01967,2.21684,0.61084;
        1.01868,2.22117,0.62776;
        1.01771,2.22548,0.64515;
        1.01676,2.22978,0.66299;
        1.01584,2.23406,0.68127;
        1.01495,2.23832,0.69994;
        1.01409,2.24257,0.71903;
        1.01326,2.24680,0.73849;
        1.01248,2.25101,0.75828;
        1.01172,2.25520,0.77839;
        1.01101,2.25938,0.79878;
        1.01033,2.26354,0.81944;
        1.00970,2.26768,0.84034;
        1.00911,2.27180,0.86145;
        1.00856,2.27591,0.88275;
        1.00805,2.28000,0.90421;
        1.00758,2.28407,0.92582;
        1.00714,2.28812,0.94753;
        1.00675,2.29216,0.96933;
        1.00639,2.29618,0.99118;
        1.00606,2.30018,1.01307;
        1.00576,2.30416,1.03496;
        1.00549,2.30812,1.05686;
        1.00525,2.31207,1.07874;
        1.00503,2.31599,1.10058;
        1.00484,2.31990,1.12237;
        1.00466,2.32378,1.14409;
        1.00451,2.32766,1.16571;
        1.00437,2.33150,1.18724;
        1.00424,2.33534,1.20865;
        1.00412,2.33915,1.22994;
        1.00401,2.34296,1.25111;
        1.00390,2.34674,1.27214;
        1.00379,2.35052,1.29304;
        1.00369,2.35427,1.31381;
        1.00358,2.35802,1.33445;
        1.00347,2.36175,1.35497;
        1.00336,2.36546,1.37537;
        1.00324,2.36917,1.39565;
        1.00312,2.37286,1.41582;
        1.00299,2.37654,1.43588;
        1.00285,2.38021,1.45585;
        1.00270,2.38387,1.47574;
        1.00254,2.38751,1.49556;
        1.00236,2.39114,1.51531;
        1.00217,2.39476,1.53503;
        1.00197,2.39837,1.55468;
        1.00175,2.40198,1.57431;
        1.00151,2.40556,1.59392;
        1.00125,2.40914,1.61352;
        1.00097,2.41271,1.63313;
        1.00068,2.41626,1.65275;
        1.00035,2.41981,1.67240;
        1.00001,2.42334,1.69208;
        0.99964,2.42686,1.71180;
        0.99925,2.43037,1.73158;
        0.99884,2.43388,1.75141;
        0.99839,2.43737,1.77131;
        0.99792,2.44085,1.79128;
        0.99743,2.44433,1.81133;
        0.99690,2.44779,1.83148;
        0.99634,2.45125,1.85172;
        0.99576,2.45470,1.87207;
        0.99515,2.45813,1.89251;
        0.99451,2.46155,1.91307;
        0.99384,2.46497,1.93374;
        0.99313,2.46837,1.95453;
        0.99240,2.47177,1.97543;
        0.99163,2.47516,1.99645;
        0.99084,2.47854,2.01756;
        0.99001,2.48191,2.03878;
        0.98915,2.48527,2.06009;
        0.98826,2.48862,2.08150;
        0.98734,2.49196,2.10300;
        0.98639,2.49529,2.12458;
        0.98540,2.49860,2.14623;
        0.98439,2.50191,2.16795;
        0.98334,2.50521,2.18972;
        0.98227,2.50850,2.21153;
        0.98117,2.51178,2.23338;
        0.98004,2.51505,2.25527;
        0.97888,2.51830,2.27718;
        0.97770,2.52155,2.29912;
        0.97649,2.52479,2.32107;
        0.97526,2.52802,2.34303;
        0.97400,2.53124,2.36500;
        0.97272,2.53445,2.38697
    ];
end

%% Function to create detailed plots for each time-stepping method
function createDetailedPlots(X, Y, u_final, u_ref, time_steps, errors_local, x, method, methodColor)
    % Setup figure with better appearance
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');
    
    % Use perceptually uniform colormap
    colormap(viridisColormap());
    
    % Plot final tumor density with better surf appearance
    subplot(2, 2, 1);
    surf_obj = surf(X, Y, u_final, 'EdgeColor', 'none');
    view(40, 30);  % Better viewing angle
    lighting gouraud;
    camlight;
    axis tight;
    title(sprintf('%s Method: Final Tumor Density', method), 'FontWeight', 'bold');
    xlabel('x-position', 'FontWeight', 'bold');
    ylabel('y-position', 'FontWeight', 'bold');
    zlabel('Density', 'FontWeight', 'bold');
    colorbar;
    
    % Plot time step sizes over time with better styling
    subplot(2, 2, 2);
    cum_time = cumsum(time_steps(1:end-1));
    plot(cum_time, time_steps(2:end), 'Color', methodColor, 'LineWidth', 2);
    title('Time Step Sizes', 'FontWeight', 'bold');
    xlabel('Simulation Time', 'FontWeight', 'bold');
    ylabel('\Delta t', 'FontWeight', 'bold');
    grid on;
    box on;
    
    % Add annotation showing total number of steps
    text(0.05, 0.95, sprintf('Total Steps: %d', length(time_steps)-1), ...
        'Units', 'normalized', 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9], ...
        'EdgeColor', 'k', 'Margin', 2);
    
    % Plot local error if available
    subplot(2, 2, 3);
    if ~isempty(errors_local)
        semilogy(cum_time, errors_local, 'Color', methodColor, 'LineWidth', 2);
        title('Local Error Estimate', 'FontWeight', 'bold');
        xlabel('Simulation Time', 'FontWeight', 'bold');
        ylabel('Error (log scale)', 'FontWeight', 'bold');
        grid on;
        box on;
        
        % Add annotation showing mean error
        mean_error = mean(errors_local);
        text(0.05, 0.95, sprintf('Mean Error: %.2e', mean_error), ...
            'Units', 'normalized', 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9], ...
            'EdgeColor', 'k', 'Margin', 2);
    else
        text(0.5, 0.5, 'Error estimation not available for this method', ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
        axis off;
    end
    
    % Plot cross-section with better styling
    subplot(2, 2, 4);
    mid_idx = floor(size(u_final, 1)/2);
    plot(x, u_final(mid_idx, :), 'Color', methodColor, 'LineWidth', 2);
    hold on;
    plot(x, u_ref(mid_idx, :), 'r--', 'LineWidth', 2);
    title('Cross-section at y = L/2', 'FontWeight', 'bold');
    xlabel('x-position', 'FontWeight', 'bold');
    ylabel('Tumor Density', 'FontWeight', 'bold');
    legend({'Computed Solution', 'Reference Solution'}, 'Location', 'best');
    grid on;
    box on;
    
    % Add annotation showing L2 error
    rel_error = norm(u_final(:) - u_ref(:)) / norm(u_ref(:));
    text(0.05, 0.95, sprintf('Relative L2 Error: %.2e', rel_error), ...
        'Units', 'normalized', 'FontWeight', 'bold', 'BackgroundColor', [0.9 0.9 0.9], ...
        'EdgeColor', 'k', 'Margin', 2);
    
    % Add overall title with method name
    sgtitle(sprintf('%s Time Stepping Method for Tumor Growth', method), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, sprintf('tumor_growth_%s.png', lower(method)));
end

%% Function to create animation of tumor growth
function createAnimation(X, Y, x, solution_snapshots, time_points, method, methodColor)
    % Setup figure with fixed size for consistent frame dimensions
    fig = figure('Position', [100, 100, 1000, 500], 'Color', 'white');
    set(fig, 'Units', 'pixels');  % Ensure units are in pixels for consistency
    
    % Create colormap
    colormap(viridisColormap());
    
    % Initialize video writer
    v = VideoWriter(sprintf('animations/tumor_growth_%s.avi', lower(method)), 'Motion JPEG AVI');
    v.FrameRate = 10;
    v.Quality = 95;
    open(v);
    
    % Get min/max values for consistent color scaling
    all_values = [];
    for i = 1:length(solution_snapshots)
        all_values = [all_values; solution_snapshots{i}(:)];
    end
    min_val = min(all_values);
    max_val = max(all_values);
    
    % Create animation frames
    for i = 1:length(solution_snapshots)
        u = solution_snapshots{i};
        t = time_points(i);
        
        % 3D Surface plot
        subplot(1, 2, 1);
        surf_obj = surf(X, Y, u, 'EdgeColor', 'none');
        view(40, 30);
        lighting gouraud;
        camlight;
        title(sprintf('Tumor Growth at t = %.2f', t), 'FontWeight', 'bold');
        xlabel('x-position', 'FontWeight', 'bold');
        ylabel('y-position', 'FontWeight', 'bold');
        zlabel('Density', 'FontWeight', 'bold');
        axis tight;
        zlim([min_val, max_val * 1.1]);
        colorbar;
        
        % Cross-section plot
        subplot(1, 2, 2);
        mid_idx = floor(size(u, 1)/2);
        plot(x, u(mid_idx, :), 'Color', methodColor, 'LineWidth', 2);
        title(sprintf('Cross-section at y = L/2, t = %.2f', t), 'FontWeight', 'bold');
        xlabel('x-position', 'FontWeight', 'bold');
        ylabel('Tumor Density', 'FontWeight', 'bold');
        grid on;
        ylim([min_val, max_val * 1.1]);
        
        % Add overall title
        sgtitle(sprintf('%s Time Stepping Method', method), 'FontSize', 16, 'FontWeight', 'bold');
        
        % Lock figure size by setting 'Position' explicitly
        set(fig, 'Position', [100, 100, 1000, 500]);
        
        % Capture frame using the figure handle
        frame = getframe(fig);
        writeVideo(v, frame);
    end
    
    % Close video file
    close(v);
    close(fig);
    
    fprintf('Animation saved to animations/tumor_growth_%s.avi\n', lower(method));
end

%% Function to create comparison plots for different methods
function createComparisonPlots(methods, errors, cputimes, methodColors)
    % Error comparison
    fig1 = figure('Position', [100, 100, 800, 500], 'Color', 'white');
    bar_handle = bar(categorical(methods), errors);
    for i = 1:length(methods)
        bar_handle.FaceColor = 'flat';
        bar_handle.CData(i,:) = methodColors{i};
    end
    title('Error Comparison Between Methods', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('Relative L2 Error', 'FontWeight', 'bold');
    grid on;
    box on;
    
    % Add error values as text
    for i = 1:length(errors)
        text(i, errors(i)/2, sprintf('%.2e', errors(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'white');
    end
    
    saveas(fig1, 'comparison_error.png');
    
    % Computation time comparison
    fig2 = figure('Position', [100, 100, 800, 500], 'Color', 'white');
    bar_handle = bar(categorical(methods), cputimes);
    for i = 1:length(methods)
        bar_handle.FaceColor = 'flat';
        bar_handle.CData(i,:) = methodColors{i};
    end
    title('Computation Time Comparison Between Methods', 'FontWeight', 'bold', 'FontSize', 14);
    ylabel('CPU Time (seconds)', 'FontWeight', 'bold');
    grid on;
    box on;
    
    % Add time values as text
    for i = 1:length(cputimes)
        text(i, cputimes(i)/2, sprintf('%.2f s', cputimes(i)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'white');
    end
    
    saveas(fig2, 'comparison_time.png');
    
    % Efficiency plot (scatterplot)
    fig3 = figure('Position', [100, 100, 800, 600], 'Color', 'white');
    
    % Create scatter plot
    for i = 1:length(methods)
        scatter(cputimes(i), errors(i), 200, methodColors{i}, 'filled', 'MarkerEdgeColor', 'k');
        hold on;
        text(cputimes(i), errors(i)*1.1, methods{i}, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center');
    end
    
    % Add a power law fit line to show ideal performance trend
    hold on;
    x_fit = linspace(min(cputimes)*0.8, max(cputimes)*1.2, 100);
    y_fit = max(errors) * (min(cputimes) ./ x_fit);
    plot(x_fit, y_fit, 'k--', 'LineWidth', 1.5);
    text(x_fit(end), y_fit(end), 'Ideal Scaling', 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
    
    title('Efficiency Comparison: Error vs. Computation Time', 'FontWeight', 'bold', 'FontSize', 14);
    xlabel('CPU Time (seconds)', 'FontWeight', 'bold');
    ylabel('Relative Error', 'FontWeight', 'bold');
    set(gca, 'YScale', 'log');
    grid on;
    box on;
    
    saveas(fig3, 'comparison_efficiency.png');
end

%% Function to create an evolution comparison plot
function createEvolutionComparisonPlot(X, Y, x, methods, time_history, solution_history, methodColors)
    % Create figure
    fig = figure('Position', [100, 100, 1200, 800], 'Color', 'white');
    
    % Select common time points
    target_times = [0, 5, 10, 15, 20];
    
    % Calculate number of rows and columns for subplots
    num_times = length(target_times);
    num_methods = length(methods);
    
    % For each time point
    for t_idx = 1:num_times
        target_time = target_times(t_idx);
        
        % For each method
        for m_idx = 1:num_methods
            time_points = time_history{m_idx};
            solution_snapshots = solution_history{m_idx};
            
            % Find closest time point
            [~, idx] = min(abs(time_points - target_time));
            u = solution_snapshots{idx};
            actual_time = time_points(idx);
            
            % Calculate subplot position
            subplot_idx = (t_idx-1)*num_methods + m_idx;
            subplot(num_times, num_methods, subplot_idx);
            
            % Create plot - contour plot for compact display
            contourf(X, Y, u, 20, 'LineStyle', 'none');
            colormap(viridisColormap());
            
            % Add labels and titles
            if t_idx == 1
                title(methods{m_idx}, 'FontWeight', 'bold');
            end
            
            if m_idx == 1
                ylabel(sprintf('t = %.1f', target_time), 'FontWeight', 'bold');
            end
            
            if t_idx == num_times
                xlabel('x-position', 'FontWeight', 'bold');
            end
            
            % Add time text
            text(0.05, 0.95, sprintf('t = %.2f', actual_time), 'Units', 'normalized', ...
                'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k', 'Margin', 2);
            
            % Set axis limits
            axis equal;
            axis tight;
            
            % Only add colorbar for rightmost plots
            if m_idx == num_methods
                colorbar('Location', 'eastoutside');
            end
        end
    end
    
    % Add overall title
    sgtitle('Evolution of Tumor Growth Over Time Across Methods', ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig, 'tumor_growth_evolution_comparison.png');
    
    % Also create a plot of mid-line cross sections for comparison
    fig2 = figure('Position', [100, 100, 1200, 500], 'Color', 'white');
    
    % Plot for each time point
    for t_idx = 1:3:num_times  % Skip some times to reduce clutter
        target_time = target_times(t_idx);
        subplot(1, ceil(num_times/3), ceil(t_idx/3));
        
        % For each method
        for m_idx = 1:num_methods
            time_points = time_history{m_idx};
            solution_snapshots = solution_history{m_idx};
            
            % Find closest time point
            [~, idx] = min(abs(time_points - target_time));
            u = solution_snapshots{idx};
            actual_time = time_points(idx);
            
            % Plot cross-section
            mid_idx = floor(size(u, 1)/2);
            plot(x, u(mid_idx, :), 'Color', methodColors{m_idx}, 'LineWidth', 2);
            hold on;
        end
        
        % Add labels and legend
        title(sprintf('t = %.1f', target_time), 'FontWeight', 'bold');
        xlabel('x-position', 'FontWeight', 'bold');
        ylabel('Tumor Density', 'FontWeight', 'bold');
        grid on;
        box on;
        
        if t_idx == 1
            legend(methods, 'Location', 'best');
        end
    end
    
    % Add overall title
    sgtitle('Cross-section Comparison Between Methods', ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    saveas(fig2, 'tumor_cross_section_comparison.png');
end