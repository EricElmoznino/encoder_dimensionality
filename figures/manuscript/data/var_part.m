% Load data
file = 'data.csv';
T = readtable(file);
ED = log10(T.ED);
MRR = T.MRR;
EncodingScore = T.EncodingScore;

% Variance partitioning
X = table(ED, MRR, 'VariableNames', {'ED', 'MRR'});
explained = regCommonality_lsqminnorm(EncodingScore, X) 
file = 'variance_partitions.csv';
writetable(explained, file, 'WriteRowNames', true);

figure
bar(explained.Percent_Total(1:3))
ylim([0 100])
xticklabels({'Unique ED', 'Unique MRR', 'Shared'})
ylabel('Percent variance')


