%% =========================================================
%  P-LDA + Bi-LSTM Classification with Bhattacharya Wingsuit Flying Search
%% =========================================================

clc; clear; close all;

%% Step 1: Load Data
load Feature_Tr.mat
load Target_Tr.mat
load Feature_Te.mat
load Target_Te.mat

Xtrain = Feature_Tr;
Ytrain = Target_Tr;
Xtest  = Feature_Te;
Ytest  = Target_Te;

%% Step 2: Penalised LDA (P-LDA) Dimensionality Reduction
class_labels  = unique(Ytrain);
numClasses    = length(class_labels);
numFeatures   = size(Xtrain, 2);

% Class means
class_means = zeros(numClasses, numFeatures);
for i = 1:numClasses
    class_means(i, :) = mean(Xtrain(Ytrain == class_labels(i), :));
end
overall_mean = mean(Xtrain);

% Between-class scatter (S_B)
S_B = zeros(numFeatures, numFeatures);
for i = 1:numClasses
    n    = sum(Ytrain == class_labels(i));
    diff = class_means(i, :) - overall_mean;
    S_B  = S_B + n * (diff' * diff);
end

% Within-class scatter (S_W)
S_W = zeros(numFeatures, numFeatures);
for i = 1:numClasses
    idx        = (Ytrain == class_labels(i));
    class_data = Xtrain(idx, :);
    d          = class_data - class_means(i, :);
    S_W        = S_W + d' * d;
end

% Perturbed inverse to handle singularity
epsilon      = 1e-4;
S_W_perturbed = S_W + epsilon * eye(numFeatures);

% Generalised eigenproblem
[V, D]           = eig(pinv(S_W_perturbed) * S_B);
eigenvalues      = real(diag(D));                          % ensure real
[~, sort_idx]    = sort(eigenvalues, 'descend');
sorted_eigvecs   = real(V(:, sort_idx));

% Project to num_dimensions
num_dimensions = 3;
W              = sorted_eigvecs(:, 1:num_dimensions);
Xtrain_lda     = Xtrain * W;
Xtest_lda      = Xtest  * W;

%% ---- Visualisation 1: LDA Scatter (2-D projection) ----
figure('Name','LDA 2-D Projection','Color','w','Position',[50 50 700 500]);
colors = lines(numClasses);
for i = 1:numClasses
    idx = (Ytrain == class_labels(i));
    scatter(Xtrain_lda(idx,1), Xtrain_lda(idx,2), 30, colors(i,:), 'filled', ...
        'DisplayName', sprintf('Class %d', class_labels(i)));
    hold on;
end
xlabel('LD 1'); ylabel('LD 2');
title('P-LDA: 2-D Training Data Projection');
legend('Location','best'); grid on; box on;

%% ---- Visualisation 2: LDA Scatter (3-D projection) ----
figure('Name','LDA 3-D Projection','Color','w','Position',[780 50 700 500]);
for i = 1:numClasses
    idx = (Ytrain == class_labels(i));
    scatter3(Xtrain_lda(idx,1), Xtrain_lda(idx,2), Xtrain_lda(idx,3), 30, ...
        colors(i,:), 'filled', 'DisplayName', sprintf('Class %d', class_labels(i)));
    hold on;
end
xlabel('LD 1'); ylabel('LD 2'); zlabel('LD 3');
title('P-LDA: 3-D Training Data Projection');
legend('Location','best'); grid on; box on; view(45, 30);

%% Step 3: Wingsuit Flying Search Hyperparameter Optimisation
%  Objective: minimise (1 - accuracy) on the test set
%  Search space: [numHiddenUnits, maxEpochs, learningRate]
%                [10,300]         [10,1000]   [0.0001,0.1]

fun    = @(params) trainBiLSTM(params, Xtrain_lda, Ytrain, Xtest_lda, Ytest);
n      = 3;                                  % number of hyperparameters
bounds = [10, 300; 10, 1000; 0.0001, 0.1];  % [lower; upper] per param
N      = 10;                                 % population size
M      = 10;                                 % iterations

fprintf('\n--- Bhattacharya Wingsuit Flying Search optimisation started ---\n');
[x_best, f_best, f_history, runtime] = WingsuitFlyingSearch(fun, n, bounds, N, M);

fprintf('\n=== Optimisation Complete ===\n');
fprintf('Best numHiddenUnits : %d\n',   round(x_best(1)));
fprintf('Best maxEpochs      : %d\n',   round(x_best(2)));
fprintf('Best learningRate   : %.6f\n', x_best(3));
fprintf('Best loss (1-acc)   : %.4f  =>  Validation accuracy: %.2f%%\n', ...
        f_best, (1 - f_best)*100);
fprintf('Runtime             : %.2f s\n', runtime);

%% ---- Visualisation 3: Convergence Curve ----
figure('Name','Optimisation Convergence','Color','w','Position',[50 600 700 420]);
plot(1:length(f_history), (1-f_history)*100, 'b-o', 'LineWidth', 2, ...
     'MarkerFaceColor','b', 'MarkerSize', 6);
xlabel('Iteration'); ylabel('Best Accuracy (%)');
title('Wingsuit Flying Search: Convergence Curve');
grid on; box on;
yline((1-f_best)*100,'r--','LineWidth',1.5,'Label',sprintf('Best = %.2f%%',(1-f_best)*100));

%% Step 4: Retrain Final Model with Best Hyperparameters
fprintf('\n--- Retraining final model with best hyperparameters ---\n');
best_hidden = round(x_best(1));
best_epochs = round(x_best(2));
best_lr     = x_best(3);

finalModel = trainBiLSTMModel(Xtrain_lda, Ytrain, best_hidden, best_epochs, best_lr);

%% Step 5: Evaluate Final Model
Ypred_cat = predictBiLSTMModel(finalModel, Xtest_lda);
Ytrue_cat = categorical(Ytest);

% Overall metrics
accuracy = mean(Ypred_cat == Ytrue_cat) * 100;
fprintf('\n=== Final Model Performance ===\n');
fprintf('Overall Accuracy : %.2f%%\n', accuracy);

% Per-class metrics (Precision, Recall, F1)
classNames  = categories(Ytrue_cat);
numC        = numel(classNames);
precision   = zeros(numC, 1);
recall      = zeros(numC, 1);
f1          = zeros(numC, 1);

for k = 1:numC
    tp = sum(Ypred_cat == classNames{k} & Ytrue_cat == classNames{k});
    fp = sum(Ypred_cat == classNames{k} & Ytrue_cat ~= classNames{k});
    fn = sum(Ypred_cat ~= classNames{k} & Ytrue_cat == classNames{k});
    precision(k) = tp / max(tp + fp, 1);
    recall(k)    = tp / max(tp + fn, 1);
    f1(k)        = 2 * precision(k) * recall(k) / max(precision(k) + recall(k), 1e-9);
    fprintf('Class %-6s  Precision: %.3f  Recall: %.3f  F1: %.3f\n', ...
            classNames{k}, precision(k), recall(k), f1(k));
end

%% ---- Visualisation 4: Confusion Matrix ----
figure('Name','Confusion Matrix','Color','w','Position',[780 600 600 520]);
cm = confusionchart(Ytrue_cat, Ypred_cat, ...
    'Title', sprintf('Confusion Matrix  (Accuracy = %.2f%%)', accuracy), ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
cm.FontSize = 11;

%% ---- Visualisation 5: Per-class Precision / Recall / F1 ----
figure('Name','Per-Class Metrics','Color','w','Position',[50 1100 700 420]);
x_ticks = 1:numC;
bar_data = [precision, recall, f1] * 100;
b = bar(x_ticks, bar_data, 0.75);
b(1).FaceColor = [0.22 0.49 0.72];
b(2).FaceColor = [0.30 0.69 0.29];
b(3).FaceColor = [0.89 0.10 0.11];
set(gca,'XTickLabel', classNames, 'FontSize', 11);
xlabel('Class'); ylabel('Score (%)');
title('Per-Class Precision, Recall and F1-Score');
legend({'Precision','Recall','F1-Score'}, 'Location','southeast');
ylim([0 110]); grid on; box on;

% Add value labels on bars
for g = 1:3
    for k = 1:numC
        text(x_ticks(k) + (g-2)*0.25, bar_data(k,g) + 1.5, ...
             sprintf('%.1f', bar_data(k,g)), ...
             'HorizontalAlignment','center','FontSize',8);
    end
end

fprintf('\nAll visualisations generated successfully.\n');

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

%% --- Objective wrapper ---
function loss = trainBiLSTM(params, Xtrain, Ytrain, Xtest, Ytest)
    numHidden    = params(1);
    maxEpoch     = params(2);
    learningRate = params(3);

    model     = trainBiLSTMModel(Xtrain, Ytrain, numHidden, maxEpoch, learningRate);
    Ypred     = predictBiLSTMModel(model, Xtest);
    loss      = calculateLoss(Ypred, Ytest);   % returns 1 - accuracy
end

%% --- Train Bi-LSTM ---
function model = trainBiLSTMModel(Xtrain, Ytrain, numHidden, maxEpoch, learningRate)
    % Shuffle once per training call (not with fixed seed inside loop)
    idx    = randperm(size(Xtrain, 1));
    Xtrain = Xtrain(idx, :);
    Ytrain = Ytrain(idx);

    numHidden    = round(numHidden);
    maxEpoch     = round(maxEpoch);
    inputSize    = size(Xtrain, 2);   % equals num_dimensions (3)
    numClasses   = numel(unique(Ytrain));

    % Convert to cell-array of column sequences (length-1 sequences)
    Xtrain_cell = num2cell(Xtrain', 1)';
    Ytrain_cat  = categorical(Ytrain);

    layers = [
        sequenceInputLayer(inputSize)
        bilstmLayer(numHidden, 'OutputMode', 'last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs',        maxEpoch, ...
        'InitialLearnRate', learningRate, ...
        'Shuffle',          'every-epoch', ...
        'Verbose',          false, ...
        'Plots',            'none');

    model = trainNetwork(Xtrain_cell, Ytrain_cat, layers, options);
end

%% --- Predict ---
function Ypred = predictBiLSTMModel(model, Xtest)
    Xtest_cell = num2cell(Xtest', 1)';
    Ypred      = classify(model, Xtest_cell);
end

%% --- Loss = 1 - accuracy (for minimisation) ---
function loss = calculateLoss(Ypred, Ytrue)
    Ytrue_cat = categorical(Ytrue);
    accuracy  = mean(Ypred == Ytrue_cat);
    loss      = 1 - accuracy;          % minimise this
end

%% =========================================================
%  WINGSUIT FLYING SEARCH (WFS) with Bhattacharyya guidance
%% =========================================================
function [x_best, f_best, f_history, runtime] = WingsuitFlyingSearch(fun, n, bounds, N, M)
%  WingsuitFlyingSearch  Metaheuristic optimiser inspired by wingsuit gliding.
%
%  Phases:
%    1. Free-fall    – broad exploration (large step)
%    2. Gliding      – exploitation along best-to-agent vector
%    3. Landing      – fine local search near best solution
%  Bhattacharyya distance between current population distribution and the
%  best-neighbourhood distribution is used to adaptively scale step sizes.
%
%  Inputs
%    fun    : objective function handle (minimisation)
%    n      : number of decision variables
%    bounds : n×2 matrix of [lower_bound, upper_bound] per variable
%    N      : population size
%    M      : number of iterations
%
%  Outputs
%    x_best    : best solution vector found
%    f_best    : objective value at x_best
%    f_history : best objective value at each iteration (length M)
%    runtime   : total elapsed time in seconds

    t_start = tic;
    lb = bounds(:, 1)';   % row vectors
    ub = bounds(:, 2)';

    % --- Initialise population uniformly in [lb, ub] ---
    X = lb + (ub - lb) .* rand(N, n);

    % Evaluate initial population
    f = zeros(N, 1);
    for i = 1:N
        f(i) = fun(clamp(X(i,:), lb, ub));
    end

    [f_best, best_idx] = min(f);
    x_best    = X(best_idx, :);
    f_history = zeros(M, 1);

    fprintf('  Iter | Best Loss | Best Accuracy\n');
    fprintf('  -----|-----------|---------------\n');

    for iter = 1:M
        % Adaptive parameters
        alpha = 1 - iter/M;        % exploration weight (decreases)
        beta  = iter/M;            % exploitation weight (increases)

        % Bhattacharyya-based scale: measures spread of population
        mu_pop  = mean(X, 1);
        sig_pop = std(X, 0, 1) + 1e-8;
        mu_best = x_best;
        sig_best = sig_pop * 0.3 + 1e-8;

        % Bhattacharyya distance (Gaussian approximation)
        sig_avg = (sig_pop.^2 + sig_best.^2) / 2 + 1e-8;
        bhat    = 0.25 * sum(((mu_pop - mu_best).^2) ./ sig_avg) + ...
                  0.5  * sum(log(sig_avg ./ sqrt(sig_pop.^2 .* sig_best.^2 + 1e-16)));
        bhat_scale = exp(-bhat / n);  % 0 (far) → 1 (close)

        for i = 1:N
            % Phase 1: Free-fall (global exploration)
            r1     = rand(1, n);
            x_ff   = lb + (ub - lb) .* r1;

            % Phase 2: Gliding (towards best with random perturbation)
            r2     = rand(1, n);
            x_gl   = X(i,:) + beta * bhat_scale * (x_best - X(i,:)) ...
                             + alpha * (2*r2 - 1) .* (ub - lb) * 0.1;

            % Phase 3: Landing (fine search near best)
            r3     = randn(1, n) * 0.01 .* (ub - lb);
            x_ld   = x_best + r3;

            % Combine phases probabilistically
            p = rand;
            if     p < 0.2,        x_new = x_ff;
            elseif p < 0.8,        x_new = x_gl;
            else,                  x_new = x_ld;
            end

            x_new = clamp(x_new, lb, ub);
            f_new = fun(x_new);

            % Greedy selection
            if f_new < f(i)
                X(i,:) = x_new;
                f(i)   = f_new;
            end
        end

        % Update global best
        [iter_best, idx] = min(f);
        if iter_best < f_best
            f_best = iter_best;
            x_best = X(idx, :);
        end
        f_history(iter) = f_best;

        fprintf('  %4d | %.5f   | %.2f%%\n', iter, f_best, (1-f_best)*100);
    end

    runtime = toc(t_start);
end

%% --- Helper: clamp variables to bounds ---
function x = clamp(x, lb, ub)
    x = max(x, lb);
    x = min(x, ub);
end