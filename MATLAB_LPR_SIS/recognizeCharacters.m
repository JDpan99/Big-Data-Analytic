function recognizedText = recognizeCharacters(plateImage)
    % Robust License Plate Character Recognition with Template Matching
    % Works with older MATLAB versions and includes character templates
    
    recognizedText = '';
    
    if isempty(plateImage)
        return;
    end
    
    fprintf('\n=== ROBUST LICENSE PLATE CHARACTER RECOGNITION ===\n');
    
    try
        % Step 1: Preprocessing - LIMITED TO 2 VARIANTS
        [preprocessedImages, imageInfo] = preprocessPlateImage(plateImage);
        
        % Step 2: Initialize character templates (if not already done)
        templates = initializeCharacterTemplates();
        
        % Step 3: Try multiple recognition approaches
        results = {};
        confidences = [];
        
        for i = 1:length(preprocessedImages)
            fprintf('\n--- Processing Image Variant %d (%s, %s polarity) ---\n', ...
                i, imageInfo(i).enhancement, ...
                iif(imageInfo(i).polarity, 'dark-on-light', 'light-on-dark'));
            
            % Method 1: Simple OCR (without language parameter)
            [text1, conf1] = trySimpleOCR(preprocessedImages{i});
            hasText1 = ~isempty(text1);
            confGood1 = conf1 > 0.3;
            if hasText1 && confGood1
                results{end+1} = text1;
                confidences(end+1) = conf1 * 1.0;
                fprintf('Simple OCR: "%s" (conf: %.3f)\n', text1, conf1);
            end
            
            % Method 2: Character Segmentation + Template Matching
            [text2, conf2] = trySegmentationTemplateMatching(preprocessedImages{i}, templates);
            hasText2 = ~isempty(text2);
            confGood2 = conf2 > 0.2;
            if hasText2 && confGood2
                results{end+1} = text2;
                confidences(end+1) = conf2 * 1.2; % Boost template matching
                fprintf('Template Matching: "%s" (conf: %.3f)\n', text2, conf2);
            end
            
            % Method 3: OCR on Segmented Characters
            [text3, conf3] = trySegmentationOCR(preprocessedImages{i});
            hasText3 = ~isempty(text3);
            confGood3 = conf3 > 0.2;
            if hasText3 && confGood3
                results{end+1} = text3;
                confidences(end+1) = conf3;
                fprintf('Segmentation OCR: "%s" (conf: %.3f)\n', text3, conf3);
            end
        end
        
        % Step 4: Select best result
        recognizedText = selectBestResult(results, confidences);
        
        fprintf('\n=== FINAL RESULT ===\n');
        fprintf('Recognized Text: "%s"\n', recognizedText);
        
        % Optional: Display debug images - FIXED: Safer conditions
        isUnknown = strcmp(recognizedText, 'UNKNOWN');
        hasImages = length(preprocessedImages) > 0;
        if isUnknown && hasImages
            showDebugImages(plateImage, preprocessedImages{1});
        end
        
    catch ME
        fprintf('Error in character recognition: %s\n', ME.message);
        recognizedText = 'ERROR';
    end
end

function [preprocessedImages, imageInfo] = preprocessPlateImage(plateImage)
    % Preprocess image with LIMITED variants for robust recognition
    
    preprocessedImages = {};
    imageInfo = struct('polarity', {}, 'enhancement', {}, 'size', {});
    
    % Convert to grayscale if needed
    if size(plateImage, 3) == 3
        grayImage = rgb2gray(plateImage);
        fprintf('Converted RGB to grayscale\n');
    else
        grayImage = plateImage;
        fprintf('Input already grayscale\n');
    end
    
    % Get image statistics
    [h, w] = size(grayImage);
    fprintf('Original size: %dx%d\n', h, w);
    
    % Resize if too small (better for both OCR and template matching)
    minHeight = 48;
    if h < minHeight
        scale = minHeight / h;
        newW = round(w * scale);
        grayImage = imresize(grayImage, [minHeight, newW], 'bicubic');
        fprintf('Resized to: %dx%d (scale: %.2f)\n', minHeight, newW, scale);
    end
    
    % Apply ONLY 2 enhancement strategies to avoid wrong results
    enhancements = {
        @(img) img,  % Original
        @(img) adapthisteq(img, 'ClipLimit', 0.02)  % CLAHE only
    };
    
    enhancementNames = {'Original', 'CLAHE'};
    
    for eIdx = 1:length(enhancements)
        enhanced = enhancements{eIdx}(grayImage);
        
        % Create both polarities for the first enhancement only
        % For second enhancement, use only one polarity
        if eIdx == 1
            polarities = createBothPolarities(enhanced);
        else
            % For CLAHE, only use the best polarity
            polarities = createBestPolarity(enhanced);
        end
        
        for pIdx = 1:length(polarities)
            preprocessedImages{end+1} = polarities{pIdx};
            
            info = struct();
            info.polarity = pIdx == 1; % true for black-on-white
            info.enhancement = enhancementNames{eIdx};
            info.size = size(polarities{pIdx});
            imageInfo(end+1) = info;
        end
        
        % Stop after 2 variants to prevent processing wrong results
        if length(preprocessedImages) >= 2
            break;
        end
    end
    
    fprintf('Created %d preprocessed variants\n', length(preprocessedImages));
end

function polarities = createBothPolarities(grayImage)
    % Create both black-on-white and white-on-black versions with smart polarity detection
    
    % Try multiple thresholding methods
    methods = {
        @(img) imbinarize(img, 'adaptive'),  % Adaptive
        @(img) imbinarize(img),              % Otsu
        @(img) imbinarize(img, 0.45),        % Fixed threshold 1
        @(img) imbinarize(img, 0.55)         % Fixed threshold 2
    };
    
    bestBinary = [];
    bestScore = 0;
    bestPolarity = true;
    
    for i = 1:length(methods)
        try
            binary = methods{i}(grayImage);
            
            % Test both polarities and their quality
            score1 = evaluateBinaryQuality(binary);
            score2 = evaluateBinaryQuality(~binary);
            
            if score1 > bestScore
                bestScore = score1;
                bestBinary = binary;
                bestPolarity = true;
            end
            
            if score2 > bestScore
                bestScore = score2;
                bestBinary = ~binary;
                bestPolarity = false;
            end
        catch
            continue;
        end
    end
    
    if isempty(bestBinary)
        bestBinary = imbinarize(grayImage);
    end
    
    % Clean the binary image
    cleaned = cleanBinaryImage(bestBinary);
    
    % Detect if this is likely a black plate with white characters
    isBlackPlate = detectBlackPlate(grayImage);
    
    if isBlackPlate
        fprintf('  Detected BLACK PLATE with white characters - adjusting polarities\n');
        % For black plates, prioritize inverted polarity (white chars on black background)
        polarities = {~cleaned, cleaned};
    else
        fprintf('  Detected LIGHT PLATE with dark characters - normal polarities\n');
        % For normal plates, prioritize normal polarity (dark chars on light background)
        polarities = {cleaned, ~cleaned};
    end
end

function polarities = createBestPolarity(grayImage)
    % Create only the best polarity version with black plate detection
    
    % Try multiple thresholding methods
    methods = {
        @(img) imbinarize(img, 'adaptive'),  % Adaptive
        @(img) imbinarize(img),              % Otsu
        @(img) imbinarize(img, 0.45),        % Fixed threshold 1
        @(img) imbinarize(img, 0.55)         % Fixed threshold 2
    };
    
    bestBinary = [];
    bestScore = 0;
    bestPolarity = true;
    
    for i = 1:length(methods)
        try
            binary = methods{i}(grayImage);
            
            % Test both polarities
            score1 = evaluateBinaryQuality(binary);
            score2 = evaluateBinaryQuality(~binary);
            
            if score1 > bestScore
                bestScore = score1;
                bestBinary = binary;
                bestPolarity = true;
            end
            
            if score2 > bestScore
                bestScore = score2;
                bestBinary = ~binary;
                bestPolarity = false;
            end
        catch
            continue;
        end
    end
    
    if isempty(bestBinary)
        bestBinary = imbinarize(grayImage);
    end
    
    % Clean the binary image
    cleaned = cleanBinaryImage(bestBinary);
    
    % Detect if this is likely a black plate with white characters
    isBlackPlate = detectBlackPlate(grayImage);
    
    if isBlackPlate
        fprintf('  Detected BLACK PLATE - using inverted polarity\n');
        % For black plates, use inverted polarity (white chars become black on white)
        polarities = {~cleaned};
    else
        fprintf('  Detected LIGHT PLATE - using normal polarity\n');
        % For normal plates, use normal polarity
        polarities = {cleaned};
    end
end

function score = evaluateBinaryQuality(binary)
    % Evaluate binary image quality for character recognition
    % FIXED: Use element-wise operations instead of logical operators
    
    % Character regions should occupy 15-45% of the image - FIXED: Scalar comparisons
    charRatio = sum(~binary(:)) / numel(binary);
    ratioTooLow = charRatio < 0.15;
    ratioTooHigh = charRatio > 0.45;
    
    if ratioTooLow || ratioTooHigh
        score = 0;
        return;
    end
    
    % Good binary should have reasonable number of connected components
    cc = bwconncomp(~binary);
    numComponents = cc.NumObjects;
    
    componentsTooFew = numComponents < 3;
    componentsTooMany = numComponents > 15;
    
    if componentsTooFew || componentsTooMany
        score = 0;
        return;
    end
    
    % Score based on how close to ideal character ratio
    idealRatio = 0.25;
    score = 1 - abs(charRatio - idealRatio) / idealRatio;
    
    % Bonus for reasonable number of components (4-8 characters expected)
    componentsGood = (numComponents >= 4) && (numComponents <= 8);
    if componentsGood
        score = score * 1.2;
    end
end

function cleaned = cleanBinaryImage(binaryImage)
    % Clean binary image while preserving character structure
    
    % Remove very small noise
    minArea = max(5, round(numel(binaryImage) * 0.001));
    cleaned = bwareaopen(~binaryImage, minArea);
    cleaned = ~cleaned;
    
    % Fill small holes
    se = strel('disk', 1);
    cleaned = imclose(cleaned, se);
    
    % Remove border objects that are likely not characters
    cleaned = imclearborder(~cleaned);
    cleaned = ~cleaned;
end

function templates = initializeCharacterTemplates()
    % Initialize character templates for template matching
    % You should replace this with actual character templates
    
    templates = struct();
    
    % For now, create placeholder templates
    % YOU SHOULD REPLACE THESE WITH ACTUAL CHARACTER IMAGES
    templates.characters = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ';
    templates.images = {};
    
    % Create simple placeholder templates (replace with real ones)
    for i = 1:length(templates.characters)
        % Create a simple 24x16 template (replace with actual character images)
        template = createSimpleCharacterTemplate(templates.characters(i));
        templates.images{i} = template;
    end
    
    fprintf('Initialized %d character templates\n', length(templates.characters));
    
    % TODO: Load actual character templates from files
    % templates = loadCharacterTemplatesFromFiles();
end

function template = createSimpleCharacterTemplate(char)
    % Create a simple template for demonstration
    % REPLACE THIS WITH YOUR ACTUAL CHARACTER TEMPLATES
    
    template = zeros(24, 16, 'logical');
    
    % This is just a placeholder - you should have actual character images
    switch char
        case '0'
            % Create a simple oval
            [X, Y] = meshgrid(1:16, 1:24);
            centerX = 8; centerY = 12;
            radiusX = 6; radiusY = 10;
            oval = ((X - centerX) / radiusX).^2 + ((Y - centerY) / radiusY).^2;
            template = (oval <= 1) & (oval >= 0.6);
            
        case '1'
            % Create a vertical line
            template(:, 7:9) = true;
            template(1:6, 6:7) = true;
            
        case 'A'
            % Create a simple A shape
            template(8:end, 7:9) = true;
            template(4:8, 5:11) = true;
            template(1:4, 7:9) = true;
            template(12:14, :) = false;
            
        otherwise
            % Default rectangle for other characters
            template(4:20, 3:13) = true;
            template(8:16, 6:10) = false;
    end
end

function [recognizedText, confidence] = trySimpleOCR(image)
    % Try OCR without language parameter (compatible with older MATLAB)
    
    recognizedText = '';
    confidence = 0;
    
    try
        % Simple OCR call without language parameter
        ocrResults = ocr(image);
        
        % Clean and validate result
        rawText = ocrResults.Text;
        cleanText = cleanOCRResult(rawText);
        
        if ~isempty(cleanText)
            % Calculate confidence - FIXED: Better error handling
            confidence = 0.5; % Default confidence
            
            if isfield(ocrResults, 'CharacterConfidences')
                charConf = ocrResults.CharacterConfidences;
                if ~isempty(charConf)
                    validConf = charConf(charConf > 0);
                    if ~isempty(validConf)
                        confidence = mean(validConf);
                    end
                end
            end
            
            % Boost confidence for reasonable lengths
            textLength = length(cleanText);
            if (textLength >= 4) && (textLength <= 8)
                confidence = confidence * 1.1;
            end
            
            % Boost confidence for Malaysian patterns
            if matchesMalaysianPattern(cleanText)
                confidence = confidence * 1.2;
            end
            
            recognizedText = cleanText;
        end
        
    catch ME
        fprintf('  Simple OCR failed: %s\n', ME.message);
    end
end

function [recognizedText, confidence] = trySegmentationTemplateMatching(image, templates)
    % Segment characters and match with templates
    
    recognizedText = '';
    confidence = 0;
    
    try
        % Find connected components (characters)
        cc = bwconncomp(~image); % Assume dark characters on light background
        if cc.NumObjects == 0
            return;
        end
        
        stats = regionprops(cc, 'BoundingBox', 'Area', 'Image');
        
        % Filter valid character candidates
        validChars = filterCharacterCandidates(stats, size(image));
        
        if isempty(validChars)
            return;
        end
        
        % Sort by x-position (left to right)
        bboxes = vertcat(validChars.BoundingBox);
        [~, sortIdx] = sort(bboxes(:, 1));
        validChars = validChars(sortIdx);
        
        % Match each character with templates
        characters = '';
        charConfidences = [];
        
        for i = 1:length(validChars)
            bbox = validChars(i).BoundingBox;
            
            % Extract and normalize character
            x1 = max(1, floor(bbox(1)));
            y1 = max(1, floor(bbox(2)));
            x2 = min(size(image, 2), ceil(bbox(1) + bbox(3)));
            y2 = min(size(image, 1), ceil(bbox(2) + bbox(4)));
            
            charImage = image(y1:y2, x1:x2);
            
            % Match with templates
            [bestChar, charConf] = matchCharacterTemplate(charImage, templates);
            
            characters = [characters, bestChar];
            charConfidences(end+1) = charConf;
        end
        
        if ~isempty(characters)
            recognizedText = characters;
            confidence = mean(charConfidences);
            
            % Boost confidence for Malaysian patterns
            if matchesMalaysianPattern(characters)
                confidence = confidence * 1.3;
            end
        end
        
    catch ME
        fprintf('  Template matching failed: %s\n', ME.message);
    end
end

function [bestChar, confidence] = matchCharacterTemplate(charImage, templates)
    % Match a character image with templates using correlation
    
    bestChar = '?';
    confidence = 0;
    bestScore = 0;
    
    try
        % Normalize character image to standard size
        standardSize = [24, 16];
        normalizedChar = imresize(~charImage, standardSize, 'nearest'); % Invert to match template
        
        % Try matching with each template
        for i = 1:length(templates.characters)
            template = templates.images{i};
            
            % Calculate normalized cross-correlation
            correlation = normxcorr2(template, normalizedChar);
            maxCorr = max(correlation(:));
            
            % Also try inverted character
            invertedChar = ~normalizedChar;
            correlationInv = normxcorr2(template, invertedChar);
            maxCorrInv = max(correlationInv(:));
            
            % Take the better correlation
            score = max(maxCorr, maxCorrInv);
            
            if score > bestScore
                bestScore = score;
                bestChar = templates.characters(i);
                confidence = score;
            end
        end
        
        % Set minimum confidence threshold
        if confidence < 0.3
            bestChar = '?';
            confidence = 0;
        end
        
    catch ME
        fprintf('    Character template matching failed: %s\n', ME.message);
        bestChar = '?';
        confidence = 0;
    end
end

function [recognizedText, confidence] = trySegmentationOCR(image)
    % Segment characters and apply OCR to each
    
    recognizedText = '';
    confidence = 0;
    
    try
        % Find connected components
        cc = bwconncomp(~image);
        if cc.NumObjects == 0
            return;
        end
        
        stats = regionprops(cc, 'BoundingBox', 'Area');
        
        % Filter valid character candidates
        validChars = filterCharacterCandidates(stats, size(image));
        
        if isempty(validChars)
            return;
        end
        
        % Sort by x-position
        bboxes = vertcat(validChars.BoundingBox);
        [~, sortIdx] = sort(bboxes(:, 1));
        validChars = validChars(sortIdx);
        
        % Recognize each character
        characters = '';
        charConfidences = [];
        
        for i = 1:length(validChars)
            bbox = validChars(i).BoundingBox;
            
            % Extract character region with padding
            pad = 2;
            x1 = max(1, floor(bbox(1)) - pad);
            y1 = max(1, floor(bbox(2)) - pad);
            x2 = min(size(image, 2), ceil(bbox(1) + bbox(3)) + pad);
            y2 = min(size(image, 1), ceil(bbox(2) + bbox(4)) + pad);
            
            charImage = image(y1:y2, x1:x2);
            
            % Resize for OCR
            if size(charImage, 1) < 20
                scale = 20 / size(charImage, 1);
                charImage = imresize(charImage, scale, 'nearest');
            end
            
            % Apply OCR to character
            try
                charOCR = ocr(charImage);
                charText = cleanOCRResult(charOCR.Text);
                
                if ~isempty(charText)
                    characters = [characters, charText(1)];
                    if isfield(charOCR, 'CharacterConfidences') && ~isempty(charOCR.CharacterConfidences)
                        charConfidences(end+1) = charOCR.CharacterConfidences(1);
                    else
                        charConfidences(end+1) = 0.5;
                    end
                else
                    characters = [characters, '?'];
                    charConfidences(end+1) = 0;
                end
            catch
                characters = [characters, '?'];
                charConfidences(end+1) = 0;
            end
        end
        
        if ~isempty(charConfidences)
            validConf = charConfidences(charConfidences > 0);
            if ~isempty(validConf)
                confidence = mean(validConf);
            end
        end
        
        recognizedText = characters;
        
    catch ME
        fprintf('  Segmentation OCR failed: %s\n', ME.message);
    end
end

function validChars = filterCharacterCandidates(stats, imageSize)
    % Filter connected components to find valid character candidates
    
    validChars = [];
    imageArea = imageSize(1) * imageSize(2);
    
    for i = 1:length(stats)
        bbox = stats(i).BoundingBox;
        area = stats(i).Area;
        
        width = bbox(3);
        height = bbox(4);
        aspectRatio = height / width;
        
        % Character validation criteria - FIXED: Separate scalar comparisons
        areaThreshLow = imageArea * 0.01;
        areaThreshHigh = imageArea * 0.4;
        validArea = (area > areaThreshLow) && (area < areaThreshHigh);
        validAspect = (aspectRatio > 0.5) && (aspectRatio < 3.5);
        validSize = (width >= 4) && (height >= 8);
        
        if validArea && validAspect && validSize
            validChars = [validChars; stats(i)];
        end
    end
end

function bestResult = selectBestResult(results, confidences)
    % Select the best recognition result with frequency-based selection
    
    bestResult = 'UNKNOWN';
    
    if isempty(results)
        return;
    end
    
    % Step 1: Count frequency of each unique result
    [uniqueResults, ~, idx] = unique(results);
    frequency = accumarray(idx, 1);
    
    % Step 2: Calculate scores for each result
    scores = zeros(size(confidences));
    
    for i = 1:length(results)
        text = results{i};
        conf = confidences(i);
        
        % Base score from confidence
        scores(i) = conf;
        
        % Length bonus
        textLength = length(text);
        if (textLength >= 6) && (textLength <= 7)
            scores(i) = scores(i) * 1.4;
        elseif (textLength >= 4) && (textLength <= 8)
            scores(i) = scores(i) * 1.2;
        end
        
        % Pattern bonus
        if matchesMalaysianPattern(text)
            scores(i) = scores(i) * 1.5;
        end
        
        % Penalize results with '?' characters
        questionCount = sum(text == '?');
        if questionCount > 0
            textLength = length(text);
            penalty = questionCount / textLength * 0.5;
            scores(i) = scores(i) * (1 - penalty);
        end
    end
    
    % Step 3: Apply frequency-based selection
    finalScores = zeros(length(uniqueResults), 1);
    avgConfidences = zeros(length(uniqueResults), 1);
    maxScores = zeros(length(uniqueResults), 1);
    
    for i = 1:length(uniqueResults)
        % Find all instances of this result
        sameResultIdx = strcmp(results, uniqueResults{i});
        sameResultScores = scores(sameResultIdx);
        sameResultConf = confidences(sameResultIdx);
        
        % Calculate metrics for this unique result
        freq = frequency(i);
        avgConf = mean(sameResultConf);
        maxScore = max(sameResultScores);
        
        % Final score combines frequency, average confidence, and max score
        % Higher frequency gets significant bonus
        frequencyBonus = 1.0;
        if freq >= 3
            frequencyBonus = 2.0;  % Strong bonus for 3+ occurrences
        elseif freq >= 2
            frequencyBonus = 1.6;  % Good bonus for 2+ occurrences
        end
        
        % Final score calculation
        finalScores(i) = maxScore * frequencyBonus * (1 + avgConf * 0.3);
        avgConfidences(i) = avgConf;
        maxScores(i) = maxScore;
        
        fprintf('Unique result "%s": freq=%d, avgConf=%.3f, maxScore=%.3f, finalScore=%.3f\n', ...
                uniqueResults{i}, freq, avgConf, maxScore, finalScores(i));
    end
    
    % Step 4: Select result with highest final score
    [maxFinalScore, bestUniqueIdx] = max(finalScores);
    
    if maxFinalScore > 0.15
        bestResult = formatMalaysianPlate(uniqueResults{bestUniqueIdx});
        selectedFreq = frequency(bestUniqueIdx);
        selectedAvgConf = avgConfidences(bestUniqueIdx);
        
        fprintf('\n--- FREQUENCY-BASED SELECTION ---\n');
        fprintf('Selected: "%s" (frequency: %d, avgConf: %.3f, finalScore: %.3f)\n', ...
                bestResult, selectedFreq, selectedAvgConf, maxFinalScore);
        
        if selectedFreq > 1
            fprintf('*** FREQUENCY WINNER: Result appeared %d times! ***\n', selectedFreq);
        end
    end
    
    % Debug output - show all individual results
    fprintf('\n--- Individual Results ---\n');
    for i = 1:length(results)
        fprintf('Result %d: "%s" (conf: %.3f, score: %.3f)\n', ...
                i, results{i}, confidences(i), scores(i));
    end
    
    % Show frequency analysis
    fprintf('\n--- Frequency Analysis ---\n');
    for i = 1:length(uniqueResults)
        fprintf('"%s": appears %d times, finalScore: %.3f\n', ...
                uniqueResults{i}, frequency(i), finalScores(i));
    end
    
    fprintf('Final Selected: "%s"\n', bestResult);
end

function cleanText = cleanOCRResult(rawText)
    % Clean OCR result text
    
    % Remove whitespace and newlines
    cleanText = regexprep(rawText, '\s+', '');
    
    % Remove non-alphanumeric characters
    cleanText = regexprep(cleanText, '[^A-Za-z0-9]', '');
    
    % Convert to uppercase
    cleanText = upper(cleanText);
    
    % Remove confusing characters that don't appear on Malaysian plates
    cleanText = regexprep(cleanText, '[IO]', ''); % Remove I and O which are often confused
end

function matches = matchesMalaysianPattern(text)
    % Check if text matches Malaysian license plate patterns
    
    matches = false;
    
    if isempty(text)
        return;
    end
    
    cleanText = regexprep(text, '[^A-Z0-9]', '');
    
    % Malaysian patterns
    patterns = {
        '^[A-Z]{3}[0-9]{4}$',      % ABC1234
        '^[A-Z][0-9]{3}[A-Z]{2}$', % A123BC
        '^[A-Z]{2}[0-9]{3}[A-Z]$', % AB123C
        '^[A-Z]{3}[0-9]{3}$',      % ABC123
        '^[A-Z]{2}[0-9]{4}$'       % AB1234
    };
    
    for i = 1:length(patterns)
        if ~isempty(regexp(cleanText, patterns{i}, 'once'))
            matches = true;
            return;
        end
    end
end

function formattedText = formatMalaysianPlate(text)
    % Format according to Malaysian license plate conventions
    
    if isempty(text) || strcmp(text, 'UNKNOWN')
        formattedText = 'UNKNOWN';
        return;
    end
    
    cleanText = regexprep(text, '[^A-Z0-9]', '');
    
    if length(cleanText) < 4
        formattedText = 'UNKNOWN';
        return;
    end
    
    % Apply Malaysian formatting - FIXED: Separate logical checks
    textLength = length(cleanText);
    
    if (textLength == 7) && all(isstrprop(cleanText(1:3), 'alpha')) && all(isstrprop(cleanText(4:7), 'digit'))
        formattedText = [cleanText(1:3), ' ', cleanText(4:7)]; % ABC 1234
    elseif textLength == 6
        if all(isstrprop(cleanText(1:3), 'alpha')) && all(isstrprop(cleanText(4:6), 'digit'))
            formattedText = [cleanText(1:3), ' ', cleanText(4:6)]; % ABC 123
        elseif all(isstrprop(cleanText(1:2), 'alpha')) && all(isstrprop(cleanText(3:6), 'digit'))
            formattedText = [cleanText(1:2), ' ', cleanText(3:6)]; % AB 1234
        else
            formattedText = cleanText;
        end
    else
        formattedText = cleanText;
    end
end

function showDebugImages(originalImage, processedImage)
    % Show debug images when recognition fails
    
    try
        figure('Name', 'Character Recognition Debug', 'Position', [100, 100, 800, 400]);
        
        subplot(1, 2, 1);
        imshow(originalImage);
        title('Original Plate Image');
        
        subplot(1, 2, 2);
        imshow(processedImage);
        title('Processed Binary Image');
        
        fprintf('Debug images displayed. Check if characters are clearly visible.\n');
        
    catch
        fprintf('Could not display debug images.\n');
    end
end

function isBlackPlate = detectBlackPlate(grayImage)
    % Detect if this is a black license plate with white characters
    
    try
        % Calculate image statistics
        meanIntensity = mean(grayImage(:));
        medianIntensity = median(grayImage(:));
        
        % Calculate histogram to understand intensity distribution
        [counts, centers] = imhist(grayImage);
        
        % Find peaks in histogram
        totalPixels = numel(grayImage);
        normalizedCounts = counts / totalPixels;
        
        % Check if most pixels are in the dark range (0-100)
        darkPixelRatio = sum(counts(1:min(100, length(counts)))) / totalPixels;
        
        % Check if most pixels are in the bright range (155-255)
        brightStart = max(1, min(155, length(counts)));
        brightPixelRatio = sum(counts(brightStart:end)) / totalPixels;
        
        % Black plate indicators:
        % 1. Low mean/median intensity (dark background)
        % 2. High percentage of dark pixels
        % 3. Bimodal distribution with dark peak dominant
        
        lowMeanIntensity = meanIntensity < 100;
        lowMedianIntensity = medianIntensity < 90;
        highDarkRatio = darkPixelRatio > 0.6;
        
        % Additional check: Look for bright characters on dark background
        % Apply simple threshold and check connectivity
        thresh = graythresh(grayImage);
        if thresh < 0.4  % Low threshold suggests dark image
            binary = imbinarize(grayImage, thresh);
            
            % Check if bright regions (potential characters) are reasonably sized
            cc = bwconncomp(binary);  % Bright regions
            if cc.NumObjects > 0
                stats = regionprops(cc, 'Area');
                areas = [stats.Area];
                reasonableSizedRegions = sum(areas > 20 & areas < numel(grayImage)*0.1);
                hasCharacterLikeRegions = reasonableSizedRegions >= 3;
            else
                hasCharacterLikeRegions = false;
            end
        else
            hasCharacterLikeRegions = false;
        end
        
        % Decision logic
        if lowMeanIntensity && lowMedianIntensity && highDarkRatio
            isBlackPlate = true;
        elseif lowMeanIntensity && hasCharacterLikeRegions
            isBlackPlate = true;
        else
            isBlackPlate = false;
        end
        
        fprintf('  Plate analysis: mean=%.1f, median=%.1f, darkRatio=%.2f, thresh=%.3f -> %s\n', ...
                meanIntensity, medianIntensity, darkPixelRatio, thresh, ...
                iif(isBlackPlate, 'BLACK PLATE', 'LIGHT PLATE'));
        
    catch ME
        fprintf('  Black plate detection failed: %s\n', ME.message);
        % Default to false (assume light plate)
        isBlackPlate = false;
    end
end

function result = iif(condition, trueValue, falseValue)
    % Inline if function
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end