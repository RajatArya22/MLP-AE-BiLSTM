function [x_min_m, f_min_m, f_vals, runtime] = WingsuitFlyingSearchWithBhattacharya(fun, n, bounds, N, M)
% -------------------------------------------------------------------------
% Wingsuit Flying Search with Bhattacharya Distance (FIXED)
%
% KEY FIXES vs original:
%   1. Bhattacharya coefficient now WEIGHTS candidate selection instead of
%      overwriting f_min_m (which was immediately undone by min(f) anyway).
%   2. Minimum population floor prevents collapse to 2-3 points.
%   3. N is no longer reduced by 2 before the loop starts.
%   4. delta_x_min guard prevents zero-step early-stop on the first iter.
% -------------------------------------------------------------------------

    % --- WFS Settings ---
    v           = 90*rand + 10;   % flier velocity  [10, 100]
    delta_x_min = zeros(1, n);    % stop only when step < this (all zeros = never)
    plot_res    = 0;

    tic;
    num_x = 1;
    % FIX 3: do NOT subtract 2 from N here — the +2 (centroid+random) is
    %         accounted for in GenerateCRP, not by pre-shrinking N.

    % Initial regular-grid step
    N0   = ceil(N^(1/n));
    step = zeros(1, n);
    for i = 1:n
        step(i) = (bounds(i,2) - bounds(i,1)) / N0;
    end

    % --- Halton quasi-random initial population ---
    halton_set = haltonset(n, 'Skip', 0, 'Leap', 0);
    halton_set = scramble(halton_set, 'RR2');
    x_offset   = ceil(rand * 1e6);
    x          = halton_set(x_offset : x_offset+N-1, :);
    for i = 1:n
        x(:,i) = x(:,i) .* (bounds(i,2) - bounds(i,1)) + bounds(i,1);
    end

    f = EvalPop(x, n, bounds, fun);

    m     = 1;
    f_max = max(f);
    a_m   = f_max;

    [f_min_m, idx] = min(f);
    x_min_m        = x(idx, :);

    [x_centr, x_rand] = GenerateCRP(x, n, f, f_min_m, a_m);
    f_extra           = EvalPop([x_centr; x_rand], n, bounds, fun);
    x = [x; x_centr; x_rand];
    f = [f; f_extra];
    [x_min_m, f_min_m] = UpdateMin([x_centr; x_rand], f_extra, x_min_m, f_min_m);
    f_vals(m) = f_min_m;

    if plot_res
        num_x = PlotRes(x, f, n, num_x, bounds, f_min_m);
    end

    delta_x1      = zeros(3, n);
    delta_x1(1,:) = step;
    delta_x1(2,:) = zeros(1, n);
    delta_x1(3,:) = -step;

    SC = CheckSC(delta_x1, delta_x_min, m, M);

    %% ---- Main Loop ----
    while ~SC
        m       = m + 1;
        alpha_m = 1 - v^(-(m-1)/(M-1));
        P_max_m = ceil(alpha_m * N);

        % FIX 2: enforce a minimum population size to prevent collapse
        MIN_POP = max(5, ceil(N * 0.2));
        N_m     = ceil(2*N / max(P_max_m, 1));
        N_m     = min(N_m, length(f));
        N_m     = max(N_m, min(MIN_POP, length(f)));   % <-- floor added

        delta_x_m = (1 - alpha_m) * delta_x1;

        % Sort ascending; keep best N_m
        [f, sort_ord] = sort(f, 'ascend');
        x = x(sort_ord, :);
        f = f(1:N_m);
        x = x(1:N_m, :);
        a_m = f(end);

        % Neighbourhood sizes
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

        % Generate neighbourhood points
        x_size   = size(x, 1);
        x_offset = ceil(rand * 1e6);

        for i = 1:N_m
            if P_m(i) < 1; continue; end

            direction = x_min_m - x(i,:);
            for j = 1:n
                if     direction(j) > 0; direction(j) =  1;
                elseif direction(j) < 0; direction(j) = -1;
                end
            end

            neighborhood = halton_set(x_offset : x_offset+P_m(i)-1, :);
            x_offset     = x_offset + P_m(i);

            x_new_i = zeros(P_m(i), n);
            for j = 1:n
                x_new_i(:,j) = (2*neighborhood(:,j) - 1 + direction(j)) ...
                               * delta_x_m(1,j) + x(i,j);
            end
            x = [x; x_new_i]; %#ok<AGROW>
        end

        % Evaluate new points
        x_new_all = x(x_size+1 : end, :);
        f_new     = EvalPop(x_new_all, n, bounds, fun);
        % ----------------------------------------------------------------
        if ~isempty(f_new) && f_min_m > 0 && sum(f_new) > eps
            % Per-point Bhattacharya coefficient against current best
            p_norm   = f_new / (sum(f_new) + eps);
            q_scalar = f_min_m / (N_m * f_min_m + eps);   % uniform dist over best
            q_norm   = repmat(q_scalar, size(f_new));
            bc_coeff = sqrt(p_norm .* q_norm);             % per-point BC term

            % Boost points whose BC coefficient suggests they are in a
            % similarly promising region (high overlap with best).
            % We slightly lower their effective fitness so the sort
            % keeps them in the elite set.
            bc_threshold = mean(bc_coeff);
            boost_mask   = bc_coeff > bc_threshold;
            f_new(boost_mask) = f_new(boost_mask) * 0.999; % tiny nudge only
        end
        % ----------------------------------------------------------------

        f = [f; f_new]; %#ok<AGROW>

        % Update global best
        [f_min_m, idx] = min(f);
        x_min_m        = x(idx, :);

        % Centroid + random
        [x_centr, x_rand] = GenerateCRP(x, n, f, f_min_m, a_m);
        f_extra           = EvalPop([x_centr; x_rand], n, bounds, fun);
        x = [x; x_centr; x_rand]; %#ok<AGROW>
        f = [f; f_extra];         %#ok<AGROW>
        [x_min_m, f_min_m] = UpdateMin([x_centr; x_rand], f_extra, x_min_m, f_min_m);

        f_vals(m) = f_min_m;
        fprintf('[WFS] Iter %d/%d | Best loss = %.4f\n', m, M, f_min_m);

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
    nPts = size(x, 1);
    f    = zeros(nPts, 1);
    for i = 1:nPts
        xi = x(i,:);
        for j = 1:n
            if xi(j) < bounds(j,1)
                xi(j) = 2*bounds(j,1) - xi(j);
                xi(j) = max(xi(j), bounds(j,1));
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
    for i = 1:length(f)
        if f(i) < f_min_m
            f_min_m = f(i);
            x_min_m = x(i,:);
        end
    end
end

function [x_centr, x_rand] = GenerateCRP(x, n, f, f_min_m, a_m)
    denom = a_m - f_min_m;
    if denom < eps
        gama = ones(length(f), 1) / length(f);
    else
        gama = 1 - (f - f_min_m) / denom;
        gama = max(gama, 0);
    end
    gama_sum = sum(gama);
    if gama_sum < eps
        gama     = ones(size(gama)) / length(gama);
        gama_sum = 1;
    end

    x_centr        = zeros(1, n);
    current_constr = zeros(n, 2);
    for i = 1:n
        x_centr(i)          = (x(:,i))' * gama / gama_sum;
        current_constr(i,:) = [min(x(:,i)), max(x(:,i))];
    end
    x_rand = ((current_constr(:,2) - current_constr(:,1)) .* rand(n,1) ...
              + current_constr(:,1))';
end

function SC = CheckSC(delta_x_m, delta_x_min, m, M)
    SC = false;
    for i = 1:length(delta_x_min)
        if delta_x_min(i) > 0 && abs(delta_x_m(1,i)) < delta_x_min(i)
            SC = true;
            disp('WFS: step below minimum — terminating.');
            return;
        end
    end
    if m >= M
        SC = true;
    end
end

function num_x = PlotRes(x, f, n, num_x, bounds, f_min_m)
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
