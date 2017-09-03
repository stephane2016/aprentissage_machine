function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

if size(z)>0
  g=z;
  for i=1:size(z)
  g(i) = 1.0 ./ (1.0 + exp(-z(i)));
  end
else
  g=1.0/(1.0+exp(-z(i)));
end

end
