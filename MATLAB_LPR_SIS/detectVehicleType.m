function vehicleType = detectVehicleType(inputImage)
    % Enhanced Vehicle Detection using MATLAB's built-in vision functions
    % Uses object detection and classification techniques
    
    vehicleType = 'Car'; % Default classification
    
    try
        % Convert to grayscale for processing
        if size(inputImage, 3) == 3
            grayImage = rgb2gray(inputImage);
        else
            grayImage = inputImage;
        end
        
        % Method 1: Use blob analysis for vehicle detection
        vehicleType = detectVehicleUsingBlobAnalysis(grayImage);
        
        % Method 2: If blob analysis is inconclusive, use feature-based detection
        if strcmp(vehicleType, 'Unknown')
            vehicleType = detectVehicleUsingFeatures(grayImage);
        end
        
        % Method 3: Final fallback using geometric analysis
        if strcmp(vehicleType, 'Unknown')
            vehicleType = detectVehicleUsingGeometry(inputImage);
        end
        
    catch ME
        fprintf('Error in vehicle type detection: %s', ME.message);
        vehicleType = 'Car'; % Safe default
    end
end

function vehicleType = detectVehicleUsingBlobAnalysis(grayImage)
    % Use MATLAB's blob analysis for vehicle detection
    
    vehicleType = 'Unknown';
    
    try
        % Preprocessing for better blob detection
        [height, width] = size(grayImage);
        
        % Resize if image is too large
        if height > 600
            scaleFactor = 600 / height;
            grayImage = imresize(grayImage, scaleFactor);
            [height, width] = size(grayImage);
        end
        
        % Edge detection using Canny (built-in, robust)
        edges = edge(grayImage, 'canny', [0.1, 0.3]);
        
        % Use morphological operations to create solid objects
        % Horizontal structuring element for vehicle bodies
        se_horiz = strel('line', 25, 0);
        se_vert = strel('line', 15, 90);
        
        % Close gaps to form vehicle shapes
        morphed = imclose(edges, se_horiz);
        morphed = imclose(morphed, se_vert);
        
        % Fill holes to create solid blobs
        filled = imfill(morphed, 'holes');
        
        % Remove small noise
        cleaned = bwareaopen(filled, 2000);
        
        % Use regionprops for blob analysis (built-in function)
        stats = regionprops(cleaned, 'Area', 'BoundingBox', 'Extent', ...
                           'Solidity', 'Perimeter', 'MajorAxisLength', ...
                           'MinorAxisLength', 'Eccentricity', 'ConvexArea');
        
        if isempty(stats)
            return;
        end
        
        % Find the largest blob (likely the main vehicle)
        [~, maxIdx] = max([stats.Area]);
        mainVehicle = stats(maxIdx);
        
        % Classify based on blob properties
        vehicleType = classifyVehicleFromBlob(mainVehicle, [height, width]);
        
    catch ME
        fprintf('Blob analysis failed: %s', ME.message);
        vehicleType = 'Unknown';
    end
end

function vehicleType = classifyVehicleFromBlob(blobStats, imageSize)
    % Classify vehicle type based on blob statistics
    
    vehicleType = 'Car'; % Default
    
    try
        bbox = blobStats.BoundingBox;
        area = blobStats.Area;
        extent = blobStats.Extent;
        solidity = blobStats.Solidity;
        eccentricity = blobStats.Eccentricity;
        
        blobWidth = bbox(3);
        blobHeight = bbox(4);
        aspectRatio = blobWidth / blobHeight;
        
        imageArea = imageSize(1) * imageSize(2);
        areaRatio = area / imageArea;
        
        % Classification rules based on vehicle characteristics
        
        % Motorcycle: Small, vertical orientation, high eccentricity
        if (aspectRatio < 1.5) && (areaRatio < 0.15) && (eccentricity > 0.7)
            vehicleType = 'Motorcycle';
            return;
        end
        
        % Bus: Large, rectangular, high extent and solidity
        if (aspectRatio > 2.2) && (areaRatio > 0.2) && (extent > 0.6) && (solidity > 0.8)
            vehicleType = 'Bus';
            return;
        end
        
        % Truck: Large, rectangular, moderate aspect ratio
        if (aspectRatio > 1.6 && aspectRatio < 2.5) && (areaRatio > 0.15) && (extent > 0.5)
            vehicleType = 'Truck';
            return;
        end
        
        % SUV: Moderate size, less elongated than sedan
        if (aspectRatio > 1.2 && aspectRatio < 2.0) && (areaRatio > 0.08) && (extent > 0.4)
            vehicleType = 'SUV';
            return;
        end
        
        % Van: Taller than wide, moderate size
        if (aspectRatio > 1.3 && aspectRatio < 2.2) && (areaRatio > 0.1) && (solidity > 0.7)
            vehicleType = 'Van';
            return;
        end
        
        % Default to Car for standard passenger vehicles
        vehicleType = 'Car';
        
    catch ME
        fprintf('Vehicle classification from blob failed: %s', ME.message);
        vehicleType = 'Car';
    end
end

function vehicleType = detectVehicleUsingFeatures(grayImage)
    % Use MATLAB's feature detection for vehicle classification
    
    vehicleType = 'Unknown';
    
    try
        % Use SURF features for vehicle analysis
        surfPoints = detectSURFFeatures(grayImage, 'MetricThreshold', 500);
        
        if surfPoints.Count == 0
            return;
        end
        
        % Extract SURF features
        [features, validPoints] = extractFeatures(grayImage, surfPoints);
        
        if isempty(features)
            return;
        end
        
        % Analyze feature distribution
        points = validPoints.Location;
        
        % Calculate feature distribution characteristics
        featureStats = analyzeFeatureDistribution(points, size(grayImage));
        
        % Classify based on feature characteristics
        vehicleType = classifyFromFeatures(featureStats);
        
    catch ME
        fprintf('Feature-based detection failed: %s', ME.message);
        vehicleType = 'Unknown';
    end
end

function featureStats = analyzeFeatureDistribution(points, imageSize)
    % Analyze the distribution of feature points
    
    featureStats = struct();
    
    try
        [height, width] = deal(imageSize(1), imageSize(2));
        
        % Basic statistics
        featureStats.numFeatures = size(points, 1);
        featureStats.density = featureStats.numFeatures / (height * width);
        
        % Spatial distribution
        x_coords = points(:, 1);
        y_coords = points(:, 2);
        
        featureStats.x_spread = max(x_coords) - min(x_coords);
        featureStats.y_spread = max(y_coords) - min(y_coords);
        featureStats.aspect_ratio = featureStats.x_spread / featureStats.y_spread;
        
        % Center of mass
        featureStats.center_x = mean(x_coords) / width;
        featureStats.center_y = mean(y_coords) / height;
        
        % Distribution in image regions
        upper_features = sum(y_coords < height/3);
        middle_features = sum(y_coords >= height/3 & y_coords < 2*height/3);
        lower_features = sum(y_coords >= 2*height/3);
        
        total_features = featureStats.numFeatures;
        if total_features > 0
            featureStats.upper_ratio = upper_features / total_features;
            featureStats.middle_ratio = middle_features / total_features;
            featureStats.lower_ratio = lower_features / total_features;
        else
            featureStats.upper_ratio = 0;
            featureStats.middle_ratio = 0;
            featureStats.lower_ratio = 0;
        end
        
        % Horizontal distribution
        left_features = sum(x_coords < width/3);
        center_features = sum(x_coords >= width/3 & x_coords < 2*width/3);
        right_features = sum(x_coords >= 2*width/3);
        
        if total_features > 0
            featureStats.left_ratio = left_features / total_features;
            featureStats.center_ratio = center_features / total_features;
            featureStats.right_ratio = right_features / total_features;
        else
            featureStats.left_ratio = 0;
            featureStats.center_ratio = 0;
            featureStats.right_ratio = 0;
        end
        
    catch ME
        fprintf('Feature analysis failed: %s', ME.message);
        % Set default values
        featureStats.numFeatures = 0;
        featureStats.density = 0;
        featureStats.aspect_ratio = 1;
    end
end

function vehicleType = classifyFromFeatures(featureStats)
    % Classify vehicle based on feature distribution
    
    vehicleType = 'Car'; % Default
    
    try
        aspectRatio = featureStats.aspect_ratio;
        density = featureStats.density;
        upperRatio = featureStats.upper_ratio;
        centerY = featureStats.center_y;
        
        % Motorcycle: Vertical orientation, features concentrated vertically
        if (aspectRatio < 1.2) && (upperRatio > 0.4)
            vehicleType = 'Motorcycle';
            return;
        end
        
        % Bus: Wide distribution, many features, features spread across image
        if (aspectRatio > 2.5) && (density > 0.0001) && (featureStats.center_ratio > 0.3)
            vehicleType = 'Bus';
            return;
        end
        
        % Truck: Wide, features often in cab area (left side for front view)
        if (aspectRatio > 1.8) && (featureStats.left_ratio > 0.4)
            vehicleType = 'Truck';
            return;
        end
        
        % Default classification based on general characteristics
        vehicleType = 'Car';
        
    catch ME
        fprintf('Feature classification failed: %s', ME.message);
        vehicleType = 'Car';
    end
end

function vehicleType = detectVehicleUsingGeometry(inputImage)
    % Final fallback using basic geometric analysis
    
    vehicleType = 'Car';
    
    try
        % Convert to grayscale if needed
        if size(inputImage, 3) == 3
            grayImage = rgb2gray(inputImage);
        else
            grayImage = inputImage;
        end
        
        [height, width] = size(grayImage);
        imageAspectRatio = width / height;
        
        % Use Hough transform to detect lines (built-in MATLAB function)
        edges = edge(grayImage, 'canny');
        [H, T, R] = hough(edges);
        
        % Find peaks in Hough transform
        peaks = houghpeaks(H, 20, 'threshold', ceil(0.3*max(H(:))));
        
        if isempty(peaks)
            return;
        end
        
        % Extract lines
        lines = houghlines(edges, T, R, peaks, 'FillGap', 50, 'MinLength', 30);
        
        % Analyze line characteristics
        horizontalLines = 0;
        verticalLines = 0;
        
        for k = 1:length(lines)
            xy = [lines(k).point1; lines(k).point2];
            
            % Calculate line angle
            angle = atan2d(xy(2,2) - xy(1,2), xy(2,1) - xy(1,1));
            angle = abs(angle);
            
            if angle < 30 || angle > 150  % Horizontal-ish
                horizontalLines = horizontalLines + 1;
            elseif angle > 60 && angle < 120  % Vertical-ish
                verticalLines = verticalLines + 1;
            end
        end
        
        % Classification based on line analysis
        totalLines = horizontalLines + verticalLines;
        if totalLines > 0
            horizontalRatio = horizontalLines / totalLines;
            
            % Motorcycle: More vertical lines
            if horizontalRatio < 0.3 && imageAspectRatio < 1.5
                vehicleType = 'Motorcycle';
            % Bus/Truck: Many horizontal lines (windows, body lines)
            elseif horizontalRatio > 0.7 && imageAspectRatio > 1.8
                if imageAspectRatio > 2.5
                    vehicleType = 'Bus';
                else
                    vehicleType = 'Truck';
                end
            % SUV: Balanced lines, moderate aspect ratio
            elseif imageAspectRatio > 1.2 && imageAspectRatio < 2.0
                vehicleType = 'SUV';
            % Default to Car
            else
                vehicleType = 'Car';
            end
        end
        
    catch ME
        fprintf('Geometric analysis failed: %s', ME.message);
        vehicleType = 'Car';
    end
end