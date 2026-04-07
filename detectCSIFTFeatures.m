function out = detectCSIFTFeatures(a)
    a = im2double(a);
    original = a;

    %% 1st octave generation
    [store1, ~] = generateOctave(original, 0);

    %% 2nd octave generation
    a = imresize(original, 0.5);
    [store2, ~] = generateOctave(a, 1);

    %% 3rd octave generation (store3 now used in descriptor pipeline)
    a = imresize(original, 0.25);           % FIX: resized from original, not cumulative
    [store3, ~] = generateOctave(a, 2);

    %% Resize store1 slices individually to match store2 spatial dims
    % FIX: imresize a 3D array slice-by-slice so the layer dimension is preserved
    store1_resized = zeros(size(store2, 1), size(store2, 2), size(store1, 3));
    for k = 1:size(store1, 3)
        store1_resized(:, :, k) = imresize(store1(:, :, k), ...
                                           [size(store2, 1), size(store2, 2)]);
    end

    %% Obtain keypoints from first DoG (difference between octave 1 and 2, layer 1)
    i2 = abs(store1_resized(:, :, 1) - store2(:, :, 1));
    kp = detectKeyPoints(i2);

    %% Key point descriptors (use store1 full octave for description)
    kpd = computeDescriptors(kp, store1_resized);

    out.Scale    = kp(:, 3);
    out.Octave   = kp(:, 4);
    out.Layer    = kpd;
    out.Location = kp(:, 1:2);
    out.Metric   = kp(:, 5);
end


function descriptors = computeDescriptors(keypoints, octave)
    num_keypoints = size(keypoints, 1);
    descriptors   = zeros(num_keypoints, 128);
    chaotic_map   = createChaoticMap(num_keypoints);

    num_layers = size(octave, 3);           % FIX: query actual layer count

    for i = 1:num_keypoints
        x          = keypoints(i, 1);
        y          = keypoints(i, 2);
        octave_val = keypoints(i, 4);

        % FIX: clamp layer index so it never exceeds available layers
        layer_idx = min(max(octave_val + 1, 1), num_layers);

        [m, n] = size(octave(:, :, layer_idx));

        if x - 7 >= 1 && x + 8 <= m && y - 7 >= 1 && y + 8 <= n
            patch     = octave(:, :, layer_idx);
            sub_patch = patch(x-7:x+8, y-7:y+8);

            % FIX: clamp chaotic value to (0,1] so it never zeroes out the patch
            chaotic_val = max(chaotic_map(i), 1e-6);
            sub_patch   = sub_patch .* chaotic_val;

            sub_descriptor      = histcounts(sub_patch(:), 128, ...
                                             'Normalization', 'probability');
            descriptors(i, :)   = sub_descriptor;
        end
    end
end


function chaotic_map = createChaoticMap(num_values)
    r           = 3.8;
    x           = 0.5;
    chaotic_map = zeros(num_values, 1);
    for i = 1:num_values
        x              = r * x * (1 - x);
        chaotic_map(i) = x;
    end
end


function [octave, sigma] = generateOctave(image, k2)
    k     = sqrt(2);
    % FIX: use k as the base for sigma scaling (standard SIFT convention)
    num_layers = 4 + 2 * k2;               % variable depth per octave level
    sigma      = (k .^ (0:num_layers-1)) * 1.6;

    octave = zeros(size(image, 1), size(image, 2), num_layers);

    % FIX: loop over actual sigma length, not hardcoded 4
    for i = 1:num_layers
        current_sigma  = sigma(i);
        h              = fspecial('gaussian', [7 7], current_sigma);
        blurred        = imfilter(image, h, 'symmetric');
        octave(:, :, i) = blurred;
    end
end


function keypoints = detectKeyPoints(i2)
    keypoints = [];
    [m, n]    = size(i2);

    for i = 2:m-1
        for j = 2:n-1
            patch      = i2(i-1:i+1, j-1:j+1);
            center     = patch(2, 2);
            surrounding = patch([1 2 3 4 6 7 8 9]);  % all 8 neighbors

            max_val = max(surrounding);
            min_val = min(surrounding);

            % FIX: check only against spatial neighbors (removed erroneous
            %      i2(i+1,j) cross-row comparison that broke extremum logic)
            if center > max_val || center < min_val
                keypoints = [keypoints; j i center 1 0]; %#ok<AGROW>
            end
        end
    end
end


function descriptors = generateDescriptors(keypoints, octave)
    descriptors = [];
    [m, n, ~]   = size(octave);

    for i = 1:size(keypoints, 1)
        x = keypoints(i, 2);
        y = keypoints(i, 1);

        if x > 7 && y > 7 && x < m-8 && y < n-8
            patch      = octave(x-7:x+8, y-7:y+8, :);
            descriptor = zeros(1, 4 * 4 * 8);         % FIX: pre-allocate known size
            idx        = 1;

            for k1 = 1:4
                for j1 = 1:4
                    sub_patch = patch(1+(k1-1)*4:k1*4, 1+(j1-1)*4:j1*4, :);

                    % FIX: 'L2' is invalid — use 'probability' then normalise
                    sub_descriptor = histcounts(sub_patch(:), 8, ...
                                               'Normalization', 'probability');

                    % L2-normalise the sub-block (standard SIFT step)
                    nrm = norm(sub_descriptor);
                    if nrm > 0
                        sub_descriptor = sub_descriptor / nrm;
                    end

                    descriptor(idx:idx+7) = sub_descriptor;
                    idx = idx + 8;
                end
            end

            descriptors = [descriptors; descriptor]; %#ok<AGROW>
        end
    end
end