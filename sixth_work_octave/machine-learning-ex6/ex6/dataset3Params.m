function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

tabC=[0.01;0.03;0.1;0.3;1;3;10;30];
tabSigma=[0.01;0.03;0.1;0.3;1;3;10;30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

tmp=100000000;
iM=0;
jM=0;


for i=1:size(tabC)
  for j=1:size(tabSigma)
  
      [model] = svmTrain(X, y, tabC(i),  @(x1, x2) gaussianKernel(x1, x2, tabSigma(j)));
      
      pred = svmPredict(model, Xval);
    
      sim= mean(double(pred~=yval));
      
      if(tmp>sim)
      tmp=sim;
      iM=i;
      jM=j;
      endif
      
  end
 end
C=tabC(iM);
sigma=tabSigma(jM);

% =========================================================================

end
