%% regCommonality_lsqminnorm
%
% Commonality coefficients for multiple linear regression
%
%% Syntax
%
% explained = regCommonality(y, X)
%
%% Description
%
% Commonality analysis to determine the unique and common explained
% variance of the variables in the multiple linear regression of y and X
% 
% * y - dependent variable in vector form
% * X - table of independent variables (only works for 2 variables)
% * explained - table of unique and common explained variance reported as R
% squared and as a percentage of the total explained variance
% 
% This code is based on the commonalityCoefficients function in the yhat
% package in R. See the following reference:
% 
% * Nimon (2013) Understanding the Results of Multiple Linear Regression- Beyond Standardized Regression Coefficients
% 
%% Example
%
%   y = modRdm(:);
%   X = table(refRdm(:), compRdm(:), 'VariableNames', {'raw', 'processed'});
%   explained = regCommonality(y, X);
%     
%% See also
%
% * regress (Matlab funciton)
%
% Michael F. Bonner | University of Pennsylvania | <http://www.michaelfbonner.com>


function explained = regCommonality_lsqminnorm(y, X)

% Independent variables
varNames = X.Properties.VariableNames;
nVars = length(varNames);

% Check number of variables
if nVars ~= 2
    error('Number of variables in X must be equal to 2. To implement this for more than two variables, try using the R function commonalityCoefficients in the yhat package.')
end

% Explained variance of full model
Xvals = table2array(X);
[~, Xvals] = pca(Xvals);  % applying PCA reduced instability of regression fits
X1 = [Xvals, ones(size(Xvals,1),1)];  % add constant
B = lsqminnorm(X1, y);
yHat = X1 * B;
r = corr(y, yHat);
Rtot = sign(r) * r.^2;

% Explained variance of individual regressors
Rvar = nan(nVars, 1);
for iVars = 1 : nVars
    Xvals = table2array(X(:,iVars));
    X1 = [Xvals, ones(size(Xvals,1),1)];  % add constant
    B = lsqminnorm(X1, y);
    yHat = X1 * B;
    r = corr(y, yHat);
    R = sign(r) * r.^2;
    Rvar(iVars) = R;    
end  % for iVars = 1 : nVars

% Unique explained variance
% * This is the explained variance of the other regressor. 
% * For example: U1 = Rtot - R2
% * See Nimon (2013) Understanding the Results of Multiple Linear Regression- Beyond Standardized Regression Coefficients
Uvar = nan(nVars, 1);
for iVars = 1 : nVars
    idxOther = ~ismember(1:nVars, iVars);  % index for other variable
    Rother = Rvar(idxOther);
    U = Rtot - Rother;
    Uvar(iVars) = U;    
end  % for iVars = 1 : nVars

% Common explained variance
C = sum(Rvar) - Rtot;

% Check that the partitions add up to the total explained variance
Rsum = sum(Uvar) + C;
% assert(Rsum==Rtot);
assert(abs(Rsum-Rtot) < 1e4*eps(min(abs(Rsum),abs(Rtot))));  % assert equality up to four decimal places

% Output table
coeffs = [Uvar; C; Rtot];
percentages = 100 * (coeffs./Rtot);
explained = table(coeffs, percentages, 'VariableNames', {'Coefficient', 'Percent_Total'}, 'RowNames',...
    {['Unique_' varNames{1}],...
    ['Unique_' varNames{2}],...
    ['Common_' varNames{1} '_and_' varNames{2}],...
    'Total'});


end  % function explained = regCommonality(y, X)