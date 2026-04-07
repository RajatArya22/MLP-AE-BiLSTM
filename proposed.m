clc
clear
close all
warning('off','all')
dbstop if error
addpath(genpath('.\SubCode\'))

%% ================= LOAD DATA =================
load Feature_Tr.mat
load Target_Tr.mat
load Feature_Te.mat
load Target_Te.mat

Xtrain = Feature_Tr;
Ytrain = Target_Tr;
Xtest  = Feature_Te;
Ytest  = Target_Te;

%% ================= STAGE 1 : P-LDA =================
class_labels = unique(Ytrain);
numClasses   = length(class_labels);

% --- Per-class means ---
class_means = zeros(numClasses, size(Xtrain, 2));
for i = 1:numClasses
    class_means(i, :) = mean(Xtrain(Ytrain == class_labels(i), :));
end
overall_mean = mean(Xtrain);

% --- Between-class scatter S_B ---
S_B = zeros(size(Xtrain, 2));
for i = 1:numClasses
    n_i  = sum(Ytrain == class_labels(i));
    diff = (class_means(i, :) - overall_mean)';
    S_B  = S_B + n_i * (diff * diff');
end

% --- Within-class scatter S_W ---
S_W = zeros(size(Xtrain, 2));
for i = 1:numClasses
    idx        = (Ytrain == class_labels(i));
    class_data = Xtrain(idx, :);
    d          = class_data - class_means(i, :);
    S_W        = S_W + d' * d;
end

% --- Perturbation for numerical stability (P-LDA) ---
epsilon       = 1e-4;
S_W_perturbed = S_W + epsilon * eye(size(S_W));

% --- Solve generalised eigenproblem ---
[V, D]        = eig(S_W_perturbed \ S_B);
eigenvalues   = real(diag(D));
[~, sort_idx] = sort(eigenvalues, 'descend');

numComponents = min(5, numClasses - 1);          % LDA max rank = C-1
W = real(V(:, sort_idx(1:numComponents)));

Xtrain_lda = Xtrain * W;                         % [N_tr  x numComponents]
Xtest_lda  = Xtest  * W;                         % [N_te  x numComponents]

%% ================= STAGE 2 : Z-SCORE NORMALISATION =================
% Fit on training set only — apply same transform to test set
[Xtrain_norm, mu, sigma] = zscore(Xtrain_lda);
Xtest_norm = (Xtest_lda - mu) ./ sigma;

%% ================= STAGE 3 : MLP AUTOENCODER =================

aeInputSize  = size(Xtrain_norm, 2);   % = numComponents (≤5)
encSizes     = [64 32];
latentSize   = 16;

layersAE = [
    featureInputLayer(aeInputSize,          'Name','ae_input')

    % ---- Encoder ----
    fullyConnectedLayer(encSizes(1),        'Name','enc_fc1')
    batchNormalizationLayer(               'Name','enc_bn1')
    reluLayer(                             'Name','enc_relu1')

    fullyConnectedLayer(encSizes(2),        'Name','enc_fc2')
    batchNormalizationLayer(               'Name','enc_bn2')
    reluLayer(                             'Name','enc_relu2')

    fullyConnectedLayer(latentSize,         'Name','enc_fc3')   % bottleneck
    reluLayer(                             'Name','enc_relu3')

    % ---- Decoder ----
    fullyConnectedLayer(encSizes(2),        'Name','dec_fc1')
    reluLayer(                             'Name','dec_relu1')

    fullyConnectedLayer(encSizes(1),        'Name','dec_fc2')
    reluLayer(                             'Name','dec_relu2')

    fullyConnectedLayer(aeInputSize,        'Name','dec_fc3')   % reconstruct input
    regressionLayer(                       'Name','ae_out')
];

optionsAE = trainingOptions('adam', ...
    'MaxEpochs',        150, ...
    'MiniBatchSize',    32,  ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle',          'every-epoch', ...
    'Verbose',          false);

% Target = input (autoencoder reconstruction)
autoencNet = trainNetwork(Xtrain_norm, Xtrain_norm, layersAE, optionsAE);

%% ================= STAGE 4 : EXTRACT ENCODER FEATURES =================

Ztrain = activations(autoencNet, Xtrain_norm, 'enc_relu3', ...
                     'OutputAs','rows');          % [N_tr  x latentSize]
Ztest  = activations(autoencNet, Xtest_norm,  'enc_relu3', ...
                     'OutputAs','rows');          % [N_te  x latentSize]

%% ================= STAGE 5 : PREPARE BiLSTM SEQUENCES =================

XtrainSeq = mat2cell(Ztrain, ones(size(Ztrain,1),1), latentSize);   % {N_tr×1}
XtrainSeq = cellfun(@(x) x', XtrainSeq, 'UniformOutput', false);    % each: [latentSize×1]

XtestSeq  = mat2cell(Ztest,  ones(size(Ztest,1),1),  latentSize);
XtestSeq  = cellfun(@(x) x', XtestSeq,  'UniformOutput', false);

YtrainCat = categorical(Ytrain);
YtestCat  = categorical(Ytest);

%% ================= STAGE 6 : BiLSTM HYPERPARAMETERS (BWFS) =================
numHiddenUnits = 203;
maxEpoch       = 352;
learningRate   = 0.047;

%% ================= STAGE 7 : BUILD BiLSTM =================
bilstmInputSize = latentSize;

layersBiLSTM = [
    sequenceInputLayer(bilstmInputSize,        'Name','seq_input')

    bilstmLayer(numHiddenUnits, ...
                'OutputMode','last',           'Name','bilstm')
    dropoutLayer(0.1,                          'Name','dropout')

    fullyConnectedLayer(numClasses,            'Name','fc_out')
    softmaxLayer(                              'Name','softmax')
    classificationLayer(                       'Name','cls_out')
];

optionsBiLSTM = trainingOptions('adam', ...
    'MaxEpochs',        maxEpoch,      ...
    'InitialLearnRate', learningRate,  ...
    'MiniBatchSize',    32,            ...
    'Shuffle',          'every-epoch', ...
    'Verbose',          false);

%% ================= STAGE 8 : TRAIN BiLSTM =================
net = trainNetwork(XtrainSeq, YtrainCat, layersBiLSTM, optionsBiLSTM);

%% ================= STAGE 9 : TEST =================
Ypred = classify(net, XtestSeq);

Actual    = double(YtestCat);
Predicted = double(Ypred);

%% ================= STAGE 10 : PERFORMANCE METRICS =================
Res = Sen_Spec_Acc1([Actual Predicted]);

disp(' ');
disp('%%%%% FINAL MODEL – P-LDA + MLP AUTOENCODER + BiLSTM %%%%%');
disp(['Accuracy    : ', num2str(Res(7))]);
disp(['Precision   : ', num2str(Res(8))]);
disp(['Recall      : ', num2str(Res(9))]);
disp(['F-Measure   : ', num2str(Res(10))]);
disp(['MCC         : ', num2str(Res(14))]);
disp(['--- Params  : nHU=', num2str(numHiddenUnits), ...
      ' | Ep=',  num2str(maxEpoch), ...
      ' | LR=',  num2str(learningRate,'%.6f'), ' ---']);