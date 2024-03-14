function D = createDistanceMatrix(delayVectors)
    n = size(delayVectors, 1);    
    D = zeros(n, n);
    for i = 1:n
        for j = 1:n
            D(i, j) = sqrt(sum((delayVectors(i, :) - delayVectors(j, :)).^2));
        end
    end
end