function [Xtrain,Ytrain,Xtest,Ytest] = divide_train_test(X,labels)
%divide_train_test Divides the input data in two sets for training and
%testing
%   The function makes sure that there is always at least 6 positive
%   samples for testing

train_ptg = 0.6;
N = length(X); 
tf = false(1,N);   
tf(1:round(train_ptg*N)) = true;  
s = RandStream('mt19937ar','Seed',2);
tf = tf(randperm(s,N));   
Xtrain = X(tf,:); 
Xtest = X(~tf,:);
Ytrain = labels(tf); 
Ytest = labels(~tf);
 
end

