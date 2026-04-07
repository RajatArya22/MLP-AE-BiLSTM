clc; clear; close all;

%% Load Data
load Feature_Tr.mat
load Target_Tr.mat
load Feature_Te.mat
load Target_Te.mat

Xtrain = Feature_Tr;
Ytrain = Target_Tr;
Xtest  = Feature_Te;
Ytest  = Target_Te;

%% -------------------- P-LDA --------------------
class_labels = unique(Ytrain);
c = length(class_labels);
d = size(Xtrain,2);

class_means = zeros(c,d);

for i = 1:c
    class_means(i,:) = mean(Xtrain(Ytrain==class_labels(i),:),1);
end

overall_mean = mean(Xtrain,1);

S_B = zeros(d);
S_W = zeros(d);

for i = 1:c
    Xi = Xtrain(Ytrain==class_labels(i),:);
    ni = size(Xi,1);

    diff = (class_means(i,:) - overall_mean)';
    S_B = S_B + ni * (diff * diff');

    Xi_centered = Xi - class_means(i,:);
    S_W = S_W + Xi_centered' * Xi_centered;
end

% perturbation
S_W = S_W + 1e-3 * eye(d);

[V,D] = eig(pinv(S_W)*S_B);

[~,idx] = sort(diag(D),'descend');
W = V(:,idx(1:min(c-1,3)));

Xtrain = Xtrain * W;
Xtest  = Xtest  * W;

%% ----------- Train / Validation Split ----------
cv = cvpartition(Ytrain,'HoldOut',0.2);

Xtr = Xtrain(training(cv),:);
Ytr = Ytrain(training(cv));

Xval = Xtrain(test(cv),:);
Yval = Ytrain(test(cv));

%% Objective Function
fun = @(p) objectiveBiLSTM(p, Xtr, Ytr, Xval, Yval);

%% Search Space (scaled properly)
% [hidden units, epochs, learning rate]
bounds = [
    20   150;     % hidden units
    20   200;     % epochs
    1e-4 1e-2     % learning rate
];

n = 3;
N = 15;   % population
M = 20;   % iterations

[x_best, f_best, curve, time] = WingsuitFlyingSearchWithBhattacharya(fun,n,bounds,N,M);

disp('Best Params:')
disp(x_best)
disp(['Best Loss: ', num2str(f_best)])

%% -------- Final Training on full data ----------
model = trainModel(Xtrain, Ytrain, x_best);

Ypred = classify(model, num2cell(Xtest',1)');

acc = mean(Ypred == categorical(Ytest));
disp(['Final Test Accuracy: ', num2str(acc)])

function loss = objectiveBiLSTM(p, Xtr, Ytr, Xval, Yval)

    % Decode parameters
    hidden = round(p(1));
    epochs = round(p(2));
    lr     = p(3);

    % Train model
    model = trainModel(Xtr, Ytr, [hidden epochs lr]);

    % Validation prediction
    Ypred = classify(model, num2cell(Xval',1)');

    Yval = categorical(Yval);

    % Use classification error
    acc = mean(Ypred == Yval);
    loss = 1 - acc;

    fprintf('H:%d E:%d LR:%f -> Acc:%f\n',hidden,epochs,lr,acc);

end

% ------------------------------------------------

function model = trainModel(X, Y, params)

    hidden = round(params(1));
    epochs = round(params(2));
    lr     = params(3);

    X = num2cell(X',1)';
    Y = categorical(Y);

    inputSize = size(X{1},1);
    numClasses = numel(categories(Y));

    layers = [
        sequenceInputLayer(inputSize)
        bilstmLayer(hidden,'OutputMode','last')
        dropoutLayer(0.3)            
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];

    options = trainingOptions('adam',...
        'MaxEpochs',epochs,...
        'InitialLearnRate',lr,...
        'MiniBatchSize',32,...
        'Shuffle','every-epoch',...
        'Verbose',false);

    model = trainNetwork(X,Y,layers,options);

end
