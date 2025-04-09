%% Adaptive Time Stepping for Tumor Growth Models
% This script implements an adaptive time stepping algorithm for a tumor growth model
% and compares different time-stepping strategies for efficiency and accuracy.

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

%% Run simulations with different time stepping methods
methods = {'Fixed', 'Adaptive', 'RK45'};
errors = zeros(length(methods), 1);
cputimes = zeros(length(methods), 1);

% Reference solution with very small time step
dt_ref = 0.001;
u_ref = simulateTumorGrowth(u0, D, r, K, alpha, dx, dt_ref, Tfinal, 'Fixed', false);

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
    [u_final, time_steps, errors_local] = simulateTumorGrowth(u0, D, r, K, alpha, dx, dt_init, Tfinal, method, true);
    cputimes(m) = toc;
    
    % Calculate error compared to reference solution
    errors(m) = norm(u_final(:) - u_ref(:)) / norm(u_ref(:));
    
    % Plot results
    figure;
    
    % Plot final tumor density
    subplot(2, 2, 1);
    surf(X, Y, u_final);
    title(sprintf('%s Method: Final Tumor Density', method));
    xlabel('x'); ylabel('y'); zlabel('Density');
    colorbar;
    
    % Plot time step sizes over time
    subplot(2, 2, 2);
    plot(cumsum(time_steps(1:end-1)), time_steps(2:end), 'b-', 'LineWidth', 1.5);
    title('Time Step Sizes');
    xlabel('Time'); ylabel('dt');
    grid on;
    
    % Plot local error if available
    if ~isempty(errors_local)
        subplot(2, 2, 3);
        semilogy(cumsum(time_steps(1:end-1)), errors_local, 'r-', 'LineWidth', 1.5);
        title('Local Error Estimate');
        xlabel('Time'); ylabel('Error (log scale)');
        grid on;
    end
    
    % Plot cross-section
    subplot(2, 2, 4);
    plot(x, u_final(N/2, :), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(x, u_ref(N/2, :), 'r--', 'LineWidth', 1.5);
    title('Cross-section at y = L/2');
    xlabel('x'); ylabel('Density');
    legend('Computed Solution', 'Reference Solution');
    grid on;
    
    sgtitle(sprintf('%s Time Stepping Method', method));
end

% Compare methods
figure;
bar(categorical(methods), errors);
title('Error Comparison');
ylabel('Relative L2 Error');

figure;
bar(categorical(methods), cputimes);
title('Computation Time Comparison');
ylabel('CPU Time (s)');

% Efficiency plot
figure;
scatter(cputimes, errors, 100, 1:length(methods), 'filled');
title('Efficiency Comparison');
xlabel('CPU Time (s)'); ylabel('Relative Error');
colormap(jet);
colorbar('Ticks', 1:length(methods), 'TickLabels', methods);
grid on;

%% Function to run tumor growth simulation with various time stepping methods
function [u, time_steps, error_estimates] = simulateTumorGrowth(u0, D, r, K, alpha, dx, dt_init, Tfinal, method, verbose)
    % Initialize
    u = u0;
    t = 0;
    dt = dt_init;
    time_steps = [0, dt];  % Store time steps
    error_estimates = [];  % For adaptive methods
    
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
        
        % Display progress
        if verbose && mod(length(time_steps), 10) == 0
            fprintf('t = %.3f (%.1f%%), dt = %.5f\n', t, 100*t/Tfinal, dt);
        end
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