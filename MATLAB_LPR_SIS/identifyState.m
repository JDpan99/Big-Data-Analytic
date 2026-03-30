function state = identifyState(plateText)
    % Identify Malaysian state / vehicle type from license plate text
    
    state = 'Unknown';
    
    if isempty(plateText) || strcmpi(plateText,'UNKNOWN') || strcmpi(plateText,'ERROR')
        return;
    end
    
    % Clean up
    cleanPlateText = upper(regexprep(plateText,'\s+',''));
    
    % --- Priority checks ---
    % Diplomatic plates (e.g. 99-64-DC)
    if ~isempty(regexp(cleanPlateText,'^\d{2}-\d{2}-DC$','once'))
        state = 'Diplomatic Corps';
        return;
    end
    
    % Military plates (Z, ZL, ZU etc.)
    if startsWith(cleanPlateText,'Z')
        if startsWith(cleanPlateText,'ZL')
            state = 'Military - Navy (Sea vehicle)';
        elseif startsWith(cleanPlateText,'ZU')
            state = 'Military - Air Force (Air vehicle)';
        else
            state = 'Military';
        end
        return;
    end
    
    % Commemorative / special (e.g. MALAYSIA, 1M4U, BAMbee, Chancellor, XIIINAM)
    commemorative = {'MALAYSIA','1M4U','BAMBEE','CHANCELLOR','XIIINAM'};
    if any(strcmp(cleanPlateText, commemorative))
        state = 'Commemorative Plate';
        return;
    end
    
    % KLIA limousines
    if startsWith(cleanPlateText,'LIMO')
        state = 'KLIA Limousine';
        return;
    end
    
    % --- Taxi plates (H + state code) ---
    if startsWith(cleanPlateText,'H')
        % Example: HWD, HJA, HBB etc.
        taxiPrefix = cleanPlateText(2:end); 
        baseState = identifyByPrefix(taxiPrefix);
        if ~strcmp(baseState,'Unknown')
            state = [baseState ' - Taxi'];
        else
            state = 'Taxi (Unknown State)';
        end
        return;
    end
    
    % --- Regular state-based prefixes ---
    state = identifyByPrefix(cleanPlateText);
end


function state = identifyByPrefix(plateText)
    % Identify state by known prefixes
    
    state = 'Unknown';
    
    % Handle multi-letter prefixes first
    if startsWith(plateText,'KV')
        state = 'Langkawi';
        return;
    end
    
    % Single-letter state prefixes
    firstChar = plateText(1);
    switch firstChar
        case 'A'
            state = 'Perak';
        case 'B'
            state = 'Selangor';
        case 'D'
            state = 'Kelantan';
        case 'F'
            state = 'Putrajaya';
        case 'J'
            state = 'Johor';
        case 'K'
            state = 'Kedah';
        case 'L'
            state = 'Labuan';
        case 'M'
            state = 'Malacca';
        case 'N'
            state = 'Negeri Sembilan';
        case 'P'
            state = 'Penang';
        case 'Q'
            state = 'Sarawak';
        case 'R'
            state = 'Perlis';
        case 'S'
            state = 'Sabah';
        case 'T'
            state = 'Terengganu';
        case {'V','W'}
            state = 'Kuala Lumpur';
        otherwise
            state = 'Unknown';
    end
end