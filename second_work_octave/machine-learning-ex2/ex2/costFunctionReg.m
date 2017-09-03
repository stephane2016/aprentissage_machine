function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[nr,nc]=size(X);
m=nr;




for i=1:m
  Jtmp=-y(i);
  
  tmp=X(i ,:)*theta;
  
  Jtmp=Jtmp*log(sigmoid(tmp));
  
  J=J+Jtmp-((1-y(i))*(log(1-sigmoid(tmp))));
  Jtmp=0;
  
  
end
tmp=0;
for i=2:size(theta)
tmp=tmp+theta(i)^2;
end
tmp=(tmp*lambda)/(2*m);
J=J/m+tmp;

 

  somme=0;
  [nr,nc]=size(grad);
   for i=1:m
      tmp=X(i ,:)*theta;
      somme=somme+(sigmoid(tmp)-y(i))*X(i,1);
    end
   grad(1)=somme/m;
   somme=0;
for j=2:nr
    for i=1:m
      tmp=X(i ,:)*theta;
      somme=somme+(sigmoid(tmp)-y(i))*X(i,j);
    end
    grad(j)=(somme+(lambda*theta(j)))/m;
    somme=0;
end

% =============================================================

end
