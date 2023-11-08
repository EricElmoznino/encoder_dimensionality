% Load data
file = 'data_snr_global.csv';
T = readtable(file);
ED = log10(T.effectiveDimensionality);
SNR = T.signal_noise_ratio_mean_;
MRR = T.MRR;
Signal = T.signal_mean_;
EncodingScore = T.score;

% Variance partitioning (encoding ~ ED,SNR)
X = table(ED, SNR, 'VariableNames', {'ED', 'SNR'});
explained = regCommonality_lsqminnorm(EncodingScore, X)
file = 'variance_partitions_encoding-ed_snr.csv';
writetable(explained, file, 'WriteRowNames', true);

% Variance partitioning (encoding ~ ED,MRR)
X = table(ED, MRR, 'VariableNames', {'ED', 'MRR'});
explained = regCommonality_lsqminnorm(EncodingScore, X)
file = 'variance_partitions_encoding-ed_mrr.csv';
writetable(explained, file, 'WriteRowNames', true);

% Variance partitioning (encoding ~ ED,Signal)
X = table(ED, Signal, 'VariableNames', {'ED', 'Signal'});
explained = regCommonality_lsqminnorm(EncodingScore, X)
file = 'variance_partitions_encoding-ed_signal.csv';
writetable(explained, file, 'WriteRowNames', true);

% Variance partitioning (MRR ~ ED,SNR)
X = table(ED, SNR, 'VariableNames', {'ED', 'SNR'});
explained = regCommonality_lsqminnorm(MRR, X)
file = 'variance_partitions_mrr-ed_snr.csv';
writetable(explained, file, 'WriteRowNames', true);

% Variance partitioning (MRR ~ ED,Signal)
X = table(ED, Signal, 'VariableNames', {'ED', 'Signal'});
explained = regCommonality_lsqminnorm(MRR, X)
file = 'variance_partitions_mrr-ed_signal.csv';
writetable(explained, file, 'WriteRowNames', true);







%%
figure
bar(explained.Percent_Total(1:3))
ylim([0 100])
xticklabels({'Unique ED', 'Unique SNR', 'Shared'})
ylabel('Percent variance')


