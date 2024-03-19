
function plot_multichan( x, y, channelNameArray )
    nChan = length(channelNameArray);
    
    % Calculate default interval for channel separation
    global amplitude_parameter;
    interval = nanmean(range(y, 2)) * nChan / amplitude_parameter;
    
    % Create vertical offsets for each channel
    y_center = linspace(-interval, interval, nChan);
    
    % % Define a colormap for the channels
    color_template = [0 100 0;
                       0 200 0;
                       0 300 0;
                       100 0 100;
                       100 0 200] * 0.001;
    c_space = repmat(color_template, [ceil(nChan/size(color_template, 1)), 1]);

    % Main plot
    chanlab = flip(channelNameArray); % Channel labels
    chanlab_pos = []; % Y-axis positions for channel labels
    lw = 1; % Line width

    % figure
    for chanIdx = 1:nChan
        shift = y_center(chanIdx) + nanmean(y(chanIdx, :), 2);
        
        plot(x, y(chanIdx, :) - shift, 'Color', c_space(chanIdx, :), 'LineWidth', lw);
        
        chanlab_pos(chanIdx) = y_center(chanIdx); % Y-axis positions for labels
        if chanIdx == 1
            hold on;
        end
    end
    hold off;

    % Enhance visibility and customize plot
    set(gca, 'YTick', chanlab_pos, 'YTickLabel', chanlab, 'Clipping', 'on', 'Box', 'off', 'LineWidth', 2);
    ylim([-1 1] * interval * 1.2); % Set Y-axis limits
    xlim([100 200])


    % Set up keyboard event handler
    % After creating the figure
    % set(gcf, 'KeyPressFcn', @keypress_callback);

    % Add X-axis label and title
    xlabel('Time (s)');
    ylabel('Channel Data');
    title('Multichannel Time-Series Data');
end