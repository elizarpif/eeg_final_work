function predictionError = CalculateError(signal, m, tau, k, W, h)
    normalizedSignal = normalizeSignal(signal);
    delayVectors = createDelayVectors(normalizedSignal, m, tau);
    distanceMatrix = createDistanceMatrix(delayVectors);
    N = size(delayVectors, 1);
    errors = zeros(N-h, 1);
    
    for i = 1:N-h

        [d, sortedIndices] = sort(distanceMatrix(i, :), 'ascend');
        nearestIndices = sortedIndices(~ismember(sortedIndices, (i-W):(i+W)) & sortedIndices <= N-h);
        nearestIndices = nearestIndices(1:k);
        
        predictions = mean(delayVectors(nearestIndices + h, 1));
        real = delayVectors(i + h, 1);
        errors(i) = (real - predictions)^2;
    end
    
    predictionError = sqrt(mean(errors));
end