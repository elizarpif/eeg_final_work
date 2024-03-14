function delayVectors = createDelayVectors(signal, m, timeDelay)
    n = (m - 1) * timeDelay;
    
    rows = length(signal) - n;

    delayVectors = zeros(rows, m);
    
    for i = 1:rows
        for j = 1:m
            index = i + n - (j - 1) * timeDelay;
            delayVectors(i, j) = signal(index);
        end
    end
end