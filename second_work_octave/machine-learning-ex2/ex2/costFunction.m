function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%theta is clos vector
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%]


[nr,nc]=size(X);
m=nr;
for i=1:m
  Jtmp=-y(i);
  
  tmp=X(i ,:)*theta;
  
  Jtmp=Jtmp*log(sigmoid(tmp));
  
  J=J+Jtmp-((1-y(i))*(log(1-sigmoid(tmp))));
  Jtmp=0;
end

J=J/m

 

  somme=0;
  [nr,nc]=size(grad);
for j=1:nr
    for i=1:m
      tmp=X(i ,:)*theta;
      somme=somme+(sigmoid(tmp)-y(i))*X(i,j);
    end
    grad(j)=somme/m;
    somme=0;
end

% =============================================================

end
