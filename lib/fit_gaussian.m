function [model] = fit_gaussian(X,Y)
%gaussian_fit_predict Generates a gaussian classification model and fits
%for given data.

% Positive Class
X_POS=X(Y==1,:);
% Negative Class
X_NEG=X(Y==0,:);
% Statistics
model.mean_pos = mean(X_POS);
model.cov_pos  = cov(X_POS);
model.mean_neg = mean(X_NEG);
model.cov_neg  = cov(X_NEG); 

end

