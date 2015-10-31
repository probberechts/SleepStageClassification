% https://www.physionet.org/physiobank/database/slpdb/
datasets = {'slp04', 'slp14', 'slp16', 'slp32', 'slp37', 'slp41', 'slp45', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x'};

for set = datasets
    db = strcat('/slpdb/', set{1});
    [siginfo,Fs] = wfdbdesc(db);
    Fs = Fs(1); % Sampling Frequency
    LengthSamples = siginfo.LengthSamples

    [tm, ECG1] = rdsamp(db, [1], 1000000);
    [~, ECG2] = rdsamp(db, [1], 2000000, 1000001);
    [~, ECG3] = rdsamp(db, [1], 3000000, 2000001);
    [~, ECG4] = rdsamp(db, [1], LengthSamples, 3000001);

    ECG = [ECG1; ECG2; ECG3; ECG4];
    
    [sample,~,~,~,~,comments]=rdann(db,'st');
    numTimeWindow = length(comments);
    timeWindow = floor(LengthSamples ./ numTimeWindow);

    % Load Sleep Stage Annotations
    labels = [];
    binarylabels = [];

    for i = 1:length(comments)
        C = char(comments{i});
        switch C(1)
            case 'W'
                S = -1;  % Subject is awake
            case 'R'
                S = 0;   % REM sleep
            case '1'
                S = 1;   % Sleep stage 1
            case '2'
                S = 2;   % Sleep stage 2
            case '3'
                S = 3;   % Sleep stage 3
            case '4'
                S = 4;   % Sleep stage 4
            otherwise
                S = -1;  % Other status
        end

        labels(i) = S;

        switch C(1)
            case 'W'
                S = -1;  % Subject is awake
            case 'R'
                S = 1;   % REM sleep
            case '1'
                S = 1;   % Sleep stage 1
            case '2'
                S = 1;   % Sleep stage 2
            case '3'
                S = 1;   % Sleep stage 3
            case '4'
                S = 1;   % Sleep stage 4
            otherwise
                S = -1;  % Other status
        end

        binarylabels(i) = S;
    end
    
    out = strcat('physionet/',set{1});
    save(out, 'ECG', 'labels', 'binarylabels')
end