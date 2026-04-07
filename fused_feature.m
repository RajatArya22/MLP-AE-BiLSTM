clc
clear
close all
warning('off','all')
dbstop if error
addpath(genpath('.\SubCode\'))

D = dir('.\Data\*.*');
inc = 1;
for i = 3:length(D)
    D1 = dir(['.\Data\',D(i).name,'\*.png']);
    for i1 = 1:length(D1)
        I = imread(['.\Data\',D(i).name,'\',D1(i1).name]);
        %% Preprocess
        I1 =histeq(I);
        se = strel('disk',5);
        I2 = imopen(I1,se);
        I3 = imclose(I2,se);
        I4 = imerode(I3,se);
        I5 = imdilate(I4,se);
        %% Feature Extraction
        points = detectCSIFTFeatures(I5);

        if size(points.Location,1)==1
            S_Feat = double([mean(points.Scale) mean(points.Octave) mean(points.Layer) (points.Location) mean(points.Metric)]);
        else
            S_Feat = double([mean(points.Scale) mean(points.Octave) mean(points.Layer) mean(points.Location) mean(points.Metric)]);
        end
        if ~isempty(points.Scale)
            hog_F = mean(extractHOGFeatures(I5));
            NDVI_F = mean2(ndvi(im2double(I5)));
            SVI_F = mean(svi(im2double(I5)));
            Kr = mean(kurtosis(im2double(I5)));
            GNDVI_F = mean(gndvi(im2double(I5)));
            Feature(inc,:) = double([S_Feat hog_F NDVI_F SVI_F Kr GNDVI_F]);
            Target(inc,1) = i-2;
            inc = inc+1
        end
    end
end

%% Data Splitting

Feature_Tr = [];   Feature_Te = []; Target_Tr = []; Target_Te = [];
K = unique(Target);
for i = 1:length(K)
    [va ind] = find(Target == K(i));
    TrP = round(length(va)*0.8);                         % Training Percentage
    LenTR(i) = TrP;  LenTE(i) = length(va)-TrP;

    Feature_Tr = [Feature_Tr ; Feature(va(1:TrP),:)];     % Training Sig
    Feature_Te = [Feature_Te ; Feature(va(TrP+1:end),:)]; % Testing Sig

    Target_Tr = [Target_Tr ; Target(va(1:TrP),1)];        % Training Target
    Target_Te = [Target_Te ; Target(va(TrP+1:end),1)];    % Testing Target
end

save Feature_Tr Feature_Tr
save Feature_Te Feature_Te
save Target_Tr Target_Tr
save Target_Te Target_Te

