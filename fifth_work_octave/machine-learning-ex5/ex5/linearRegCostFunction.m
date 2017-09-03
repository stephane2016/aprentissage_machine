function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

tmp=X*theta;
tmp1=tmp.-y;
tmp=tmp1.^2;
tmp=sum(tmp);
tmp=tmp+lambda*(sum(theta.^2)-(theta(1).^2));
tmp=tmp/(2*m);
J=tmp;

grad(1)=(sum(tmp1.*X(:, 1)))/m;
for j=2:size(theta)
  grad(j)=(sum(tmp1.*X(:, j))+lambda*theta(j))/m;
end



% =========================================================================

grad = grad(:);

end
