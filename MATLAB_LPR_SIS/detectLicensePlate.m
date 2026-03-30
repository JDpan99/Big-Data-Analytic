function [detectedPlate, plateRegion] = detectLicensePlate(inputImage)
    % Flexible License Plate Detection for various distances and sizes
    % Focuses on essential plate characteristics while being size-adaptive

    detectedPlate = [];
    plateRegion = [];

    try
        % Convert to grayscale if needed
        if size(inputImage, 3) == 3
            grayImage = rgb2gray(inputImage);
        else
            grayImage = inputImage;
        end
        
        [imgHeight, imgWidth] = size(grayImage);
        
        % Adaptive size thresholds based on image size
        minPlateArea = round(imgHeight * imgWidth * 0.001); % 0.1% of image
        maxPlateArea = round(imgHeight * imgWidth * 0.15);  % 15% of image
        minWidth = max(20, round(imgWidth * 0.02));         % 2% of image width
        maxWidth = round(imgWidth * 0.4);                   % 40% of image width
        minHeight = max(8, round(imgHeight * 0.01));        % 1% of image height
        maxHeight = round(imgHeight * 0.2);                 % 20% of image height
        
        fprintf('Adaptive thresholds - Area: [%d, %d], Width: [%d, %d], Height: [%d, %d]\n', ...
                minPlateArea, maxPlateArea, minWidth, maxWidth, minHeight, maxHeight);
        
        % Step 1: Enhanced preprocessing
        enhancedImage = enhanceImageForPlateDetection(grayImage);
        
        % Step 2: Multiple edge detection strategies
        edgeMaps = createMultipleEdgeMaps(enhancedImage);
        
        % Step 3: Find candidates using different approaches
        allCandidates = [];
        
        for edgeIdx = 1:length(edgeMaps)
            candidates = findPlateCandidatesFromEdges(edgeMaps{edgeIdx}, grayImage, ...
                minPlateArea, maxPlateArea, minWidth, maxWidth, minHeight, maxHeight);
            allCandidates = [allCandidates; candidates];
        end
        
        % Step 4: Remove duplicate candidates (same region detected multiple times)
        allCandidates = removeDuplicateCandidates(allCandidates);
        
        if isempty(allCandidates)
            % Try with more relaxed constraints
            fprintf('No candidates found, trying with relaxed constraints...\n');
            minPlateArea = round(minPlateArea * 0.3);
            minWidth = round(minWidth * 0.5);
            minHeight = round(minHeight * 0.5);
            
            for edgeIdx = 1:length(edgeMaps)
                candidates = findPlateCandidatesFromEdges(edgeMaps{edgeIdx}, grayImage, ...
                    minPlateArea, maxPlateArea, minWidth, maxWidth, minHeight, maxHeight);
                allCandidates = [allCandidates; candidates];
            end
            allCandidates = removeDuplicateCandidates(allCandidates);
        end
        
        if isempty(allCandidates)
            disp('No valid license plate candidates found.');
            visualizeDetectionProcess(inputImage, grayImage, enhancedImage, edgeMaps, []);
            return;
        end
        
        % Step 5: Score and rank candidates
        for i = 1:size(allCandidates, 1)
            bbox = allCandidates(i, 5:8);
            score = scorePlateCandidate(bbox, grayImage, inputImage);
            allCandidates(i, 4) = score;
        end
        
        % Sort by score
        [~, sortIdx] = sort(allCandidates(:, 4), 'descend');
        allCandidates = allCandidates(sortIdx, :);
        
        % Step 6: Validate top candidates
        for candIdx = 1:min(5, size(allCandidates, 1))
            bbox = allCandidates(candIdx, 5:8);
            score = allCandidates(candIdx, 4);
            
            % Extract region
            x1 = max(1, floor(bbox(1)));
            y1 = max(1, floor(bbox(2)));
            x2 = min(imgWidth, floor(bbox(1) + bbox(3)));
            y2 = min(imgHeight, floor(bbox(2) + bbox(4)));
            
            plateRegion = [x1, y1, x2-x1, y2-y1];
            candidatePlate = inputImage(y1:y2, x1:x2, :);
            candidateGray = grayImage(y1:y2, x1:x2);
            
            % Simple validation - not too strict
            if isReasonablePlate(candidateGray, candidatePlate)
                detectedPlate = candidatePlate;
                fprintf('License plate detected (candidate %d, score: %.3f)\n', candIdx, score);
                break;
            end
        end
        
        % Visualization
        visualizeDetectionProcess(inputImage, grayImage, enhancedImage, edgeMaps, allCandidates);
        if ~isempty(detectedPlate)
            figure; imshow(detectedPlate); title('Final Detected Plate');
        end
        
    catch ME
        fprintf('Error in license plate detection: %s\n', ME.message);
        detectedPlate = [];
        plateRegion = [];
    end
end

function enhancedImage = enhanceImageForPlateDetection(grayImage)
    % Enhanced preprocessing focusing on text visibility
    
    % Apply Gaussian blur to reduce noise
    smoothed = imgaussfilt(grayImage, 0.5);
    
    % Enhance local contrast
    enhanced = adapthisteq(smoothed, 'ClipLimit', 0.01);
    
    % Sharpen to make text more distinct
    enhancedImage = imsharpen(enhanced, 'Radius', 1, 'Amount', 0.5);
end

function edgeMaps = createMultipleEdgeMaps(image)
    % Create multiple edge detection results for robustness
    
    edgeMaps = {};
    
    % Method 1: Canny with different thresholds
    edgeMaps{1} = edge(image, 'Canny', [0.05, 0.15]);
    edgeMaps{2} = edge(image, 'Canny', [0.1, 0.25]);
    edgeMaps{3} = edge(image, 'Canny', [0.15, 0.35]);
    
    % Method 2: Sobel
    edgeMaps{4} = edge(image, 'Sobel');
    
    % Method 3: Roberts
    edgeMaps{5} = edge(image, 'Roberts');
    
    % Method 4: Combined approach
    combinedEdges = edgeMaps{1} | edgeMaps{2} | edgeMaps{4};
    edgeMaps{6} = combinedEdges;
end

function candidates = findPlateCandidatesFromEdges(edgeImage, grayImage, ...
    minArea, maxArea, minWidth, maxWidth, minHeight, maxHeight)
    % Find plate candidates from edge image with flexible constraints
    
    candidates = [];
    
    % Apply morphological operations to connect text characters
    se_horizontal = strel('rectangle', [1, 8]);  % Connect horizontally
    se_close = strel('rectangle', [2, 4]);       % Fill small gaps
    
    % Morphological closing to connect characters
    morphed = imclose(edgeImage, se_horizontal);
    morphed = imclose(morphed, se_close);
    
    % Fill holes and remove small objects
    filled = imfill(morphed, 'holes');
    cleaned = bwareaopen(filled, max(50, round(minArea * 0.1)));
    
    % Find connected components
    [labeledImage, numRegions] = bwlabel(cleaned);
    
    if numRegions == 0
        return;
    end
    
    stats = regionprops(labeledImage, 'BoundingBox', 'Area', 'Extent', 'Solidity');
    
    for i = 1:numRegions
        bbox = stats(i).BoundingBox;
        width = bbox(3);
        height = bbox(4);
        area = stats(i).Area;
        extent = stats(i).Extent;
        solidity = stats(i).Solidity;
        
        % Calculate aspect ratio
        aspectRatio = width / height;
        
        % Flexible constraints for different plate sizes
        validAspectRatio = (aspectRatio >= 1.2 && aspectRatio <= 8.0);
        validArea = (area >= minArea && area <= maxArea);
        validWidth = (width >= minWidth && width <= maxWidth);
        validHeight = (height >= minHeight && height <= maxHeight);
        validExtent = (extent >= 0.3); % Not too sparse
        validSolidity = (solidity >= 0.3); % Not too fragmented
        
        if validAspectRatio && validArea && validWidth && validHeight && validExtent && validSolidity
            % Basic score based on shape characteristics
            score = calculateBasicScore(aspectRatio, area, extent, solidity, minArea, maxArea);
            candidates = [candidates; i, aspectRatio, area, score, bbox];
        end
    end
end

function score = calculateBasicScore(aspectRatio, area, extent, solidity, minArea, maxArea)
    % Calculate basic score for initial filtering
    
    % Aspect ratio score (prefer 2:1 to 5:1, but be flexible)
    if aspectRatio >= 2.0 && aspectRatio <= 5.0
        aspectScore = 1.0;
    elseif aspectRatio >= 1.5 && aspectRatio <= 7.0
        aspectScore = 0.7;
    else
        aspectScore = 0.4;
    end
    
    % Area score (prefer middle range)
    idealArea = (minArea + maxArea) / 2;
    areaScore = 1.0 / (1.0 + abs(area - idealArea) / idealArea);
    
    % Shape quality scores
    extentScore = extent;
    solidityScore = solidity;
    
    % Combined score
    score = aspectScore * 0.4 + areaScore * 0.3 + extentScore * 0.2 + solidityScore * 0.1;
end

function uniqueCandidates = removeDuplicateCandidates(candidates)
    % Remove candidates that represent the same region
    
    if size(candidates, 1) <= 1
        uniqueCandidates = candidates;
        return;
    end
    
    uniqueCandidates = [];
    used = false(size(candidates, 1), 1);
    
    for i = 1:size(candidates, 1)
        if used(i)
            continue;
        end
        
        bbox1 = candidates(i, 5:8);
        bestCandidate = candidates(i, :);
        
        % Check for overlapping candidates
        for j = i+1:size(candidates, 1)
            if used(j)
                continue;
            end
            
            bbox2 = candidates(j, 5:8);
            
            % Calculate overlap
            overlap = calculateBboxOverlap(bbox1, bbox2);
            
            if overlap > 0.5 % Significant overlap
                used(j) = true;
                % Keep the one with better score
                if candidates(j, 4) > bestCandidate(4)
                    bestCandidate = candidates(j, :);
                end
            end
        end
        
        uniqueCandidates = [uniqueCandidates; bestCandidate];
        used(i) = true;
    end
end

function overlap = calculateBboxOverlap(bbox1, bbox2)
    % Calculate overlap ratio between two bounding boxes
    
    x1_1 = bbox1(1); y1_1 = bbox1(2); x2_1 = bbox1(1) + bbox1(3); y2_1 = bbox1(2) + bbox1(4);
    x1_2 = bbox2(1); y1_2 = bbox2(2); x2_2 = bbox2(1) + bbox2(3); y2_2 = bbox2(2) + bbox2(4);
    
    % Calculate intersection
    x1_int = max(x1_1, x1_2);
    y1_int = max(y1_1, y1_2);
    x2_int = min(x2_1, x2_2);
    y2_int = min(y2_1, y2_2);
    
    if x2_int <= x1_int || y2_int <= y1_int
        overlap = 0;
        return;
    end
    
    intersection = (x2_int - x1_int) * (y2_int - y1_int);
    area1 = bbox1(3) * bbox1(4);
    area2 = bbox2(3) * bbox2(4);
    union = area1 + area2 - intersection;
    
    overlap = intersection / union;
end

function score = scorePlateCandidate(bbox, grayImage, colorImage)
    % Score candidate based on plate-like characteristics
    
    % Extract region
    x1 = max(1, floor(bbox(1)));
    y1 = max(1, floor(bbox(2)));
    x2 = min(size(grayImage, 2), floor(bbox(1) + bbox(3)));
    y2 = min(size(grayImage, 1), floor(bbox(2) + bbox(4)));
    
    regionGray = grayImage(y1:y2, x1:x2);
    regionColor = colorImage(y1:y2, x1:x2, :);
    
    if isempty(regionGray) || size(regionGray, 1) < 5 || size(regionGray, 2) < 10
        score = 0;
        return;
    end
    
    % Feature 1: Edge density (text has reasonable edge density)
    edges = edge(regionGray, 'Canny');
    edgeDensity = sum(edges(:)) / numel(edges);
    edgeScore = 0;
    if edgeDensity > 0.05 && edgeDensity < 0.5
        edgeScore = min(1.0, edgeDensity * 10);
    end
    
    % Feature 2: Intensity variation (text creates variation)
    intensityVar = std2(regionGray) / 128;
    varScore = min(1.0, intensityVar * 2);
    
    % Feature 3: Aspect ratio preference
    aspectRatio = bbox(3) / bbox(4);
    if aspectRatio >= 2.0 && aspectRatio <= 5.0
        aspectScore = 1.0;
    elseif aspectRatio >= 1.5 && aspectRatio <= 7.0
        aspectScore = 0.7;
    else
        aspectScore = 0.3;
    end
    
    % Feature 4: Size preference (relative to image)
    imgArea = size(grayImage, 1) * size(grayImage, 2);
    regionArea = bbox(3) * bbox(4);
    sizeRatio = regionArea / imgArea;
    
    if sizeRatio >= 0.005 && sizeRatio <= 0.08 % 0.5% to 8% of image
        sizeScore = 1.0;
    elseif sizeRatio >= 0.001 && sizeRatio <= 0.15 % Extended range
        sizeScore = 0.6;
    else
        sizeScore = 0.2;
    end
    
    % Feature 5: Color uniformity (plates have relatively uniform background)
    colorScore = analyzeRegionColors(regionColor);
    
    % Combined score
    score = edgeScore * 0.25 + varScore * 0.20 + aspectScore * 0.25 + sizeScore * 0.20 + colorScore * 0.10;
end

function colorScore = analyzeRegionColors(regionColor)
    % Analyze color characteristics of the region
    
    colorScore = 0.5; % Default neutral score
    
    if size(regionColor, 3) ~= 3 || numel(regionColor) < 100
        return;
    end
    
    % Convert to different color spaces
    r = double(regionColor(:,:,1));
    g = double(regionColor(:,:,2));
    b = double(regionColor(:,:,3));
    
    % Calculate mean colors
    meanR = mean(r(:)) / 255;
    meanG = mean(g(:)) / 255;
    meanB = mean(b(:)) / 255;
    
    % Check for common plate colors
    % White/light colored plates
    if meanR > 0.6 && meanG > 0.6 && meanB > 0.6
        whiteness = min([meanR, meanG, meanB]);
        colorScore = 0.5 + whiteness * 0.5;
    end
    
    % Dark/black plates
    if meanR < 0.4 && meanG < 0.4 && meanB < 0.4
        darkness = 1 - max([meanR, meanG, meanB]);
        colorScore = 0.5 + darkness * 0.5;
    end
    
    % Yellow plates
    if meanR > 0.6 && meanG > 0.6 && meanB < 0.4
        yellowness = (meanR + meanG) / 2 - meanB;
        colorScore = 0.5 + yellowness * 0.5;
    end
end

function isReasonable = isReasonablePlate(plateGray, plateColor)
    % Simple validation - not too strict
    
    isReasonable = false;
    
    try
        [h, w] = size(plateGray);
        
        % Basic size check
        if h < 8 || w < 15
            return;
        end
        
        % Check aspect ratio
        aspectRatio = w / h;
        if aspectRatio < 1.0 || aspectRatio > 10.0
            return;
        end
        
        % Check if there's some variation (indicating text/patterns)
        variation = std2(plateGray);
        if variation < 5 % Too uniform, likely not a plate with text
            return;
        end
        
        % Check for reasonable edge content
        edges = edge(plateGray, 'Canny');
        edgeDensity = sum(edges(:)) / numel(edges);
        if edgeDensity < 0.02 || edgeDensity > 0.6
            return;
        end
        
        % If we get here, it's reasonable
        isReasonable = true;
        
        fprintf('Plate validation - Size: %dx%d, Aspect: %.2f, Variation: %.1f, Edges: %.3f - PASS\n', ...
                w, h, aspectRatio, variation, edgeDensity);
        
    catch
        isReasonable = false;
    end
end

function visualizeDetectionProcess(inputImage, grayImage, enhancedImage, edgeMaps, candidates)
    % Visualize the detection process for debugging
    
    figure('Position', [50, 50, 1400, 900]);
    
    % Original and processed images
    subplot(3,4,1); imshow(inputImage); title('Original Image');
    subplot(3,4,2); imshow(grayImage); title('Grayscale');
    subplot(3,4,3); imshow(enhancedImage); title('Enhanced');
    
    % Edge detection results
    if length(edgeMaps) >= 3
        subplot(3,4,4); imshow(edgeMaps{1}); title('Canny Edges 1');
        subplot(3,4,5); imshow(edgeMaps{2}); title('Canny Edges 2');
        subplot(3,4,6); imshow(edgeMaps{4}); title('Sobel Edges');
    end
    
    % Morphological processing example
    if ~isempty(edgeMaps)
        se = strel('rectangle', [1, 8]);
        morphed = imclose(edgeMaps{1}, se);
        filled = imfill(morphed, 'holes');
        subplot(3,4,7); imshow(morphed); title('After Morphology');
        subplot(3,4,8); imshow(filled); title('After Fill');
    end
    
    % Candidates visualization
    subplot(3,4,9:12);
    imshow(inputImage);
    hold on;
    
    if ~isempty(candidates)
        colors = ['r', 'g', 'b', 'y', 'm', 'c'];
        for i = 1:min(6, size(candidates, 1))
            bbox = candidates(i, 5:8);
            score = candidates(i, 4);
            color = colors(mod(i-1, 6) + 1);
            rectangle('Position', bbox, 'EdgeColor', color, 'LineWidth', 2);
            text(bbox(1), bbox(2)-10, sprintf('#%d: %.3f', i, score), ...
                 'Color', color, 'FontSize', 10, 'FontWeight', 'bold');
        end
        title(sprintf('All Candidates (%d found)', size(candidates, 1)));
    else
        title('No Candidates Found');
    end
end