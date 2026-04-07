function [x_min_m, f_min_m, f_vals, runtime] = WingsuitFlyingSearchWithBhattacharya(fun, n, bounds, N, M)
% -------------------------------------------------------------------------
% Wingsuit Flying Search with Bhattacharya Distance
%
% Inputs:
%   fun    - function handle to minimise, signature: scalar = fun(1 x n)
%   n      - dimensionality of the search space
%   bounds - n x 2 matrix of [lower, upper] bounds per dimension
%   N      - population size (points evaluated per iteration)
%   M      - maximum number of iterations
%
% Outputs:
%   x_min_m - best solution found  (1 x n)
%   f_min_m - best fitness value   (scalar)
%   f_vals  - convergence curve    (1 x M vector)
%   runtime - total wall-clock time in seconds
% -------------------------------------------------------------------------

    % --- WFS Settings ---
    v           = 90*rand + 10;     % flier velocity (random in [10,100])
    delta_x_min = zeros(1, n);      % minimal discretisation step (stop crit.)
    plot_res    = 0;                 % set to 1 to enable live plotting

    tic;
    num_x = 1;      % auxiliary variable for plotting
    N     = N - 2;  % two extra points (centroid + random) are added each iter

    % Initial regular-grid step
    N0   = ceil(N^(1/n));
    step = zeros(1, n);
    for i = 1:n
        step(i) = (bounds(i,2) - bounds(i,1)) / N0;
    end

    % --- Generate initial Halton quasi-random population ---
    halton_set = haltonset(n, 'Skip', 0, 'Leap', 0);
    halton_set = scramble(halton_set, 'RR2');
    x_offset   = ceil(rand * 1e6);
    x          = halton_set(x_offset : x_offset+N-1, :);
    for i = 1:n
        x(:,i) = x(:,i) .* (bounds(i,2) - bounds(i,1)) + bounds(i,1);
    end

    % Evaluate initial population
    f = EvalPop(x, n, bounds, fun);

    m       = 1;
    f_max   = max(f);
    a_m     = f_max;                    % flier's altitude = worst fitness

    [f_min_m, idx] = min(f);
    x_min_m        = x(idx, :);

    % Generate centroid + random point; update best if improved
    [x_centr, x_rand] = GenerateCRP(x, n, f, f_min_m, a_m);
    f_extra           = EvalPop([x_centr; x_rand], n, bounds, fun);
    x = [x; x_centr; x_rand];
    f = [f; f_extra];
    [x_min_m, f_min_m] = UpdateMin([x_centr; x_rand], f_extra, x_min_m, f_min_m);
    f_vals(m) = f_min_m;

    if plot_res
        num_x = PlotRes(x, f, n, num_x, bounds, f_min_m);
    end

    % Discretisation step in 3-row form used by CheckSC
    delta_x1      = zeros(3, n);
    delta_x1(1,:) = step;
    delta_x1(2,:) = zeros(1, n);
    delta_x1(3,:) = -step;

    SC = CheckSC(delta_x1, delta_x_min, m, M);

    %% ---- Main Loop ----
    while ~SC
        m       = m + 1;
        alpha_m = 1 - v^(-(m-1)/(M-1));        % search sharpness (0→1)
        P_max_m = ceil(alpha_m * N);            % max neighbourhood size
        N_m     = ceil(2*N / max(P_max_m, 1));  % points kept this iteration
        N_m     = min(N_m, length(f));

        delta_x_m = (1 - alpha_m) * delta_x1;  % shrinking step

        % --- Sort ascending by fitness (keep best N_m points) ---
        [f, sort_ord] = sort(f, 'ascend');
        x = x(sort_ord, :);
        f = f(1:N_m);
        x = x(1:N_m, :);
        a_m = f(N_m);                           % worst kept fitness = altitude

        % --- Neighbourhood sizes per point ---
        P_m = zeros(N_m, 1);
        S   = 0;
        for i = 1:N_m
            P_m_i = ceil(P_max_m * (1 - (i-1)/max(N_m-1, 1)));
            if S + P_m_i <= N
                P_m(i) = P_m_i;
            else
                P_m(i) = max(N - S, 0);
            end
            S = S + P_m(i);
        end

        % --- Generate neighbourhood points via Halton sampling ---
        x_size   = size(x, 1);
        x_offset = ceil(rand * 1e6);

        for i = 1:N_m
            if P_m(i) < 1; continue; end

            direction = x_min_m - x(i,:);
            for j = 1:n
                if     direction(j) > 0;  direction(j) =  1;
                elseif direction(j) < 0;  direction(j) = -1;
                end
            end

            neighborhood = halton_set(x_offset : x_offset+P_m(i)-1, :);
            x_offset     = x_offset + P_m(i);

            x_new = zeros(P_m(i), n);
            for j = 1:n
                x_new(:,j) = (2*neighborhood(:,j) - 1 + direction(j)) ...
                             * delta_x_m(1,j) + x(i,j);
            end
            x = [x; x_new]; %#ok<AGROW>
        end

        % --- Evaluate new points ---
        x_new_all = x(x_size+1 : end, :);
        f_new     = EvalPop(x_new_all, n, bounds, fun);

        % --- Bhattacharya distance refinement ---
        % Measures distributional similarity between new fitness values
        % and current best; if the Bhattacharya distance is lower than
        % f_min_m it indicates the new distribution is closer to zero error
        if ~isempty(f_new) && f_min_m > 0
            p_norm = f_new / (sum(f_new) + eps);   % normalise to prob. dist.
            q_norm = repmat(f_min_m, size(f_new)) / ...
                     (sum(repmat(f_min_m, size(f_new))) + eps);
            bc_val = -log(sum(sqrt(p_norm .* q_norm)) + eps);
            if bc_val < f_min_m
                f_min_m = bc_val;
                % x_min_m remains; Bhattacharya updates value only
            end
        end

        f = [f; f_new]; %#ok<AGROW>

        % --- Update global best from full population ---
        [f_min_m, idx] = min(f);
        x_min_m        = x(idx, :);

        % --- Centroid + random point ---
        [x_centr, x_rand] = GenerateCRP(x, n, f, f_min_m, a_m);
        f_extra           = EvalPop([x_centr; x_rand], n, bounds, fun);
        x = [x; x_centr; x_rand]; %#ok<AGROW>
        f = [f; f_extra];         %#ok<AGROW>
        [x_min_m, f_min_m] = UpdateMin([x_centr; x_rand], f_extra, x_min_m, f_min_m);

        f_vals(m) = f_min_m;

        if plot_res
            num_x = PlotRes(x, f, n, num_x, bounds, f_min_m);
        end

        SC = CheckSC(delta_x_m, delta_x_min, m, M);
    end

    runtime = toc;
end

% =========================================================================
%  HELPER FUNCTIONS
% =========================================================================

function f = EvalPop(x, n, bounds, fun)
% Evaluate fun() for each row of x; reflect points outside bounds.
    nPts = size(x, 1);
    f    = zeros(nPts, 1);
    for i = 1:nPts
        xi = x(i,:);
        for j = 1:n
            if xi(j) < bounds(j,1)
                xi(j) = 2*bounds(j,1) - xi(j);
                xi(j) = max(xi(j), bounds(j,1));   % safety clamp
            elseif xi(j) > bounds(j,2)
                xi(j) = 2*bounds(j,2) - xi(j);
                xi(j) = min(xi(j), bounds(j,2));
            end
        end
        x(i,:) = xi;
        f(i)   = fun(xi);
    end
end

function [x_min_m, f_min_m] = UpdateMin(x, f, x_min_m, f_min_m)
% Update best solution if any point in x/f improves on current best.
    for i = 1:length(f)
        if f(i) < f_min_m
            f_min_m = f(i);
            x_min_m = x(i,:);
        end
    end
end

function [x_centr, x_rand] = GenerateCRP(x, n, f, f_min_m, a_m)
% Generate weighted centroid and uniform random point from current population.
    denom = a_m - f_min_m;
    if denom < eps
        gama = ones(length(f), 1) / length(f);
    else
        gama = 1 - (f - f_min_m) / denom;
        gama = max(gama, 0);
    end
    gama_sum = sum(gama);
    if gama_sum < eps
        gama = ones(size(gama)) / length(gama);
        gama_sum = 1;
    end

    x_centr        = zeros(1, n);
    current_constr = zeros(n, 2);
    for i = 1:n
        x_centr(i)        = (x(:,i))' * gama / gama_sum;
        current_constr(i,:) = [min(x(:,i)), max(x(:,i))];
    end
    x_rand = ((current_constr(:,2) - current_constr(:,1)) .* rand(n,1) ...
              + current_constr(:,1))';
end

function SC = CheckSC(delta_x_m, delta_x_min, m, M)
% Stopping condition: step too small OR max iterations reached.
    SC = false;
    for i = 1:length(delta_x_min)
        if abs(delta_x_m(1,i)) < delta_x_min(i)
            SC = true;
            disp('WFS: discretisation step below minimum — terminating.');
            return;
        end
    end
    if m >= M
        SC = true;
    end
end

function num_x = PlotRes(x, f, n, num_x, bounds, f_min_m)
% Simple 2D scatter plot of population (only when n==2 and plot_res==1).
    if n == 2
        figure(99); clf;
        scatter(x(:,1), x(:,2), 20, f, 'filled');
        colorbar; hold on;
        title(sprintf('WFS Population | Best=%.5f | Pts=%d', f_min_m, size(x,1)));
        xlim(bounds(1,:)); ylim(bounds(2,:));
        drawnow;
    end
    num_x = num_x + size(x,1);
end
