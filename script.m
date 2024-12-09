% ------------------------------------------------------------------- %
% script for walking period detection
% ------------------------------------------------------------------- %

clear;

% list up CSV file
%folderPath = 'C:\Users\owner\OneDrive\デスクトップ\parkin\neck_pj\Healthy'; 
folderPath = 'C:\Users\owner\OneDrive\デスクトップ\parkin\neck_pj\Parkin'; 
csvFiles = dir(fullfile(folderPath, '*.csv'));
fnum = size(csvFiles,1);

% Table for save
T = table('Size',[fnum 5], 'VariableTypes',{'string','double','double','double','double'}, ...
          'VariableNames', {'fname','WS','TS','TE','WE'});

for i=1:fnum

    % load each file
    currentFileName = csvFiles(i).name;
    fullPath = fullfile(folderPath, currentFileName);
    dataTable = readtable(fullPath);
    splitted = strsplit(currentFileName, {'_', '.'});
    key_name1 = splitted{1};
    key_name2 = splitted{2};

    % gen vec (with filtering)
    vi = my_gen_cec(dataTable{:,1}, dataTable{:,2}, dataTable{:,3});

    % change detection (two points)
    TF = ischange(vi, 'variance', 'MaxNumChanges', 2);
    index = find(TF==1);
    T.fname(i) = currentFileName;
    T.WS(i) = index(1);
    T.WE(i) = index(2);

    % plot
    h = figure();
    plot(vi); hold on;
    len = size(vi,1);
    wline = min(min(vi)).*ones(1,len);
    wline(index(1):index(2)) = max(max(vi));
    plot(wline);

    % save plots
    sname = fullfile(folderPath, ['plot_', key_name1,'_', key_name2, '.png']);
    saveas(h, sname);
    close(h);

end
disp(T);

% ------------------------------------------------------------------- %
% cal Vector
% ------------------------------------------------------------------- %
function [vec] = my_gen_cec(wx, wy, wz)
wx = wx - my_lowpass(wx);
wy = wy - my_lowpass(wy);
wz = wz - my_lowpass(wz);
d_wx = detrend(wx,'constant');
d_wy = detrend(wy,'constant');
d_wz = detrend(wz,'constant');
vec = [d_wx, d_wy, d_wz];
end

% ------------------------------------------------------------------- %
% filtering: low-pass
% ------------------------------------------------------------------- %
function [fw] = my_lowpass(w)

% set paramters
dt = 0.01;        % sampling interval
fc2 = 0.3;       % cut-off frequency
fs = 1/dt;        % sampling frequency
order = 4;

% band-pass filtering for noise removal
[b, a] = butter(order, fc2/(fs/2), 'low');
fw = filtfilt(b, a, w);

end



