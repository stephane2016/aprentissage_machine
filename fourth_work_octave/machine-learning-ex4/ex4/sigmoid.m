function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
g=z;
[m,n]=size(g);
for i=1:m
  for j=1:n
g(i,j) = 1.0 ./ (1.0 + exp(-g(i,j)));
  end
end



end