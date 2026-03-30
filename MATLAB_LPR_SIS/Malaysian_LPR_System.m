function Malaysian_LPR_System
    % Malaysian License Plate Recognition and State Identification System
    % Main GUI Application
    
    % Create main figure
    fig = figure('Name', 'Malaysian LPR & State Identification System', ...
                'Position', [100, 100, 1200, 800], ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'Resize', 'off');
    
    % Initialize variables
    currentImage = [];
    processedImage = [];
    detectedPlate = [];
    recognizedText = '';
    identifiedState = '';
    vehicleType = '';
    
    % Create UI components
    [statusText, plateText, stateText, vehicleText] = createUI();
    
    function [statusText, plateText, stateText, vehicleText] = createUI()
        % Title
        uicontrol('Style', 'text', ...
                 'String', 'Malaysian License Plate Recognition System', ...
                 'Position', [350, 750, 500, 30], ...
                 'FontSize', 16, ...
                 'FontWeight', 'bold');
        
        % Load Image Button
        uicontrol('Style', 'pushbutton', ...
                 'String', 'Load Image', ...
                 'Position', [50, 700, 100, 30], ...
                 'Callback', @loadImage);
        
        % Process Image Button
        uicontrol('Style', 'pushbutton', ...
                 'String', 'Process Image', ...
                 'Position', [160, 700, 100, 30], ...
                 'Callback', @processImage);
        
        % Clear Results Button
        uicontrol('Style', 'pushbutton', ...
                 'String', 'Clear All', ...
                 'Position', [270, 700, 100, 30], ...
                 'Callback', @clearResults);
        
        % Original Image Display
        axes('Position', [0.05, 0.4, 0.4, 0.35]);
        title('Original Image');
        
        % Processed Image Display
        axes('Position', [0.55, 0.4, 0.4, 0.35]);
        title('Detected License Plate');
        
        % Results Panel
        uicontrol('Style', 'text', ...
                 'String', 'Recognition Results:', ...
                 'Position', [50, 300, 200, 20], ...
                 'FontSize', 12, ...
                 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'left');
        
        % Results display
        uicontrol('Style', 'text', ...
                 'String', 'License Plate:', ...
                 'Position', [50, 270, 100, 20], ...
                 'HorizontalAlignment', 'left');
        
        plateText = uicontrol('Style', 'text', ...
                             'String', 'None', ...
                             'Position', [160, 270, 200, 20], ...
                             'HorizontalAlignment', 'left', ...
                             'BackgroundColor', 'white');
        
        uicontrol('Style', 'text', ...
                 'String', 'State:', ...
                 'Position', [50, 240, 100, 20], ...
                 'HorizontalAlignment', 'left');
        
        stateText = uicontrol('Style', 'text', ...
                             'String', 'None', ...
                             'Position', [160, 240, 200, 20], ...
                             'HorizontalAlignment', 'left', ...
                             'BackgroundColor', 'white');
        
        uicontrol('Style', 'text', ...
                 'String', 'Vehicle Type:', ...
                 'Position', [50, 210, 100, 20], ...
                 'HorizontalAlignment', 'left');
        
        vehicleText = uicontrol('Style', 'text', ...
                               'String', 'None', ...
                               'Position', [160, 210, 200, 20], ...
                               'HorizontalAlignment', 'left', ...
                               'BackgroundColor', 'white');
        
        % Status text
        statusText = uicontrol('Style', 'text', ...
                              'String', 'Ready - Please load an image', ...
                              'Position', [50, 50, 500, 20], ...
                              'HorizontalAlignment', 'left');
    end
    
    function loadImage(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.gif', ...
                                         'Image Files (*.jpg,*.jpeg,*.png,*.bmp,*.gif)'}, ...
                                         'Select an image');
        if filename ~= 0
            currentImage = imread(fullfile(pathname, filename));
            
            % Display original image
            figure(fig);
            subplot(2, 2, 1);
            imshow(currentImage);
            title('Original Image');
            
            set(statusText, 'String', 'Image loaded successfully. Click Process Image to continue.');
        end
    end
    
    function processImage(~, ~)
    if isempty(currentImage)
        set(statusText, 'String', 'Please load an image first.');
        return;
    end
    
    set(statusText, 'String', 'Processing image...');
    drawnow;
    
    try
        % Step 1: Vehicle Type Detection
        vehicleType = detectVehicleType(currentImage);
        
        % Step 2: License Plate Detection
        [detectedPlate, plateRegion] = detectLicensePlate(currentImage);
        
        % Debugging step: Show intermediate results
        if isempty(detectedPlate)
            set(statusText, 'String', 'No license plate detected in the image.');
            return;
        else
            disp('Plate detected! Proceeding to character recognition...');
        end
        
        % Step 3: Character Recognition
        recognizedText = recognizeCharacters(detectedPlate);
        
        % Step 4: State Identification
        identifiedState = identifyState(recognizedText);
        
        % Display results
        if ~isempty(detectedPlate)
            figure(fig);
            subplot(2, 2, 2);
            imshow(detectedPlate);
            title('Detected License Plate');
            
            % Update text fields
            set(plateText, 'String', recognizedText);
            set(stateText, 'String', identifiedState);
            set(vehicleText, 'String', vehicleType);
            
            set(statusText, 'String', sprintf('Processing complete! Plate: %s, State: %s, Vehicle: %s', ...
                                            recognizedText, identifiedState, vehicleType));
        else
            set(statusText, 'String', 'No license plate detected in the image.');
        end
        
    catch ME
        set(statusText, 'String', ['Error: ' ME.message]);
    end
end

    
    function clearResults(~, ~)
        % Clear all variables and displays
        currentImage = [];
        processedImage = [];
        detectedPlate = [];
        recognizedText = '';
        identifiedState = '';
        vehicleType = '';
        
        % Clear displays
        figure(fig);
        subplot(2, 2, 1);
        cla;
        title('Original Image');
        
        subplot(2, 2, 2);
        cla;
        title('Detected License Plate');
        
        % Reset text fields
        set(plateText, 'String', 'None');
        set(stateText, 'String', 'None');
        set(vehicleText, 'String', 'None');
        set(statusText, 'String', 'Ready - Please load an image');
    end
end