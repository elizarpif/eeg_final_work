function normalizedSignal = normalizeSignal(signal)
    normalizedSignal = (signal - mean(signal)) / std(signal);
end