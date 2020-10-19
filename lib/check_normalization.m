function [correct] = check_normalization(features_n)
%check_normalization Checks that normalization was correctly implemented
m = sum(mean(features_n,1));
s = sum(std(features_n)) / size(features_n,2);
mean_holds = m < 1e-6;
std_holds = s < 1.0001 && s > 0.999;
correct = mean_holds && std_holds;
if correct
    disp('Normalization performed correctly!')
else
    disp('Something went wrong! Incorrect Normalization')
end

end