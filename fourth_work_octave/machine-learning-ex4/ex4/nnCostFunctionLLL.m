function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);


         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
    X=[ones((m),1) X];
    layer1=Theta1*X';
    
    z2=layer1=layer1';
    
   
    a2=layer1=sigmoid(layer1);
    
    
    layer1= [ones(size(layer1,1),1)  layer1];
    layer1=layer1';
    layer2=Theta2*layer1;
    layer2;
    layer2=sigmoid(layer2);
    
     p=max(layer2, [], 1);
      
    for i=1:m
      for j=1:num_labels
        if (p(i)==layer2(j,i))
        p(i)=j;
        end
      end
    end
    
    
%fin preparation output   h(theta) == layer2
[nr,nc]=size(X);
m=nr;
n=nc;


  un=ones(size(y),num_labels);
  Jtmp=zeros(size(y),num_labels);

 for i=1:m
 Jtmp(i,y(i))=1;
 end
  back=Jtmp;
  yTmp=Jtmp;
  Jtmp=-Jtmp;
  tmp=layer2;
  
  Jtmp=log(tmp).*Jtmp';
  
  
  
  J=sum(Jtmp'.-((un.-yTmp).*(log(un.-tmp'))));
  
 % tmp1=sum(theta.^2)-(theta(1).^2);
  %tmp1=(tmp1*lambda)/(2*m);
  
  J=J/m;
  J=sum(J);
  tmp=double((lambda/(2*m)));
  
  Theta1=Theta1';
  Theta2=Theta2';
  
  tmp=tmp*((sum((sum((double(Theta1(2:end ,:).^2)))))+(sum((sum(double(Theta2(2:end ,:).^2)))))));
  
  J=J+tmp;
tmp=0;



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%ATENTION SI PETIT ECART SUREMENT TU AS OUBLIER DE NE PAS TRAITER UN THERME EN PLACE 1 COMME THETA(1)
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

delta_3=zeros(size(Theta2));
delta_2=zeros(size(Theta1));

    Theta1=Theta1';
    Theta2=Theta2';
Jtmp=Jtmp';



for t = 1:m
    % Forward to calculate error for sample t
    
   
    
    a1 = X(t,:);
    
    
    
   
    
   
    tmpZ2 = Theta1 *a1';
    
    tmpA2 = sigmoid(tmpZ2);
    
   
    tmpA2 = [1; tmpA2];
    
    tmpZ3=Theta2*tmpA2;
    tmpA3 = sigmoid(tmpZ3);
    
    
    
    
    % Error
    
    
    yt = Jtmp(t,:);
    
    delta_3 = tmpA3 - yt';
    %juste
    %zurruck
     
    delta_2=Theta2' *delta_3.* sigmoidGradient([1; tmpZ2]);
    
   
    delta_2=delta_2(2:end);

   % printf('tableau d2 a1 a2 delta_3\');
    %delta_2 tmpA2 tmpA3 delta_3
    DELTA2 = delta_3 * tmpA2';
    
    tmpA2=tmpA2(2:end);
    
    DELTA1 =delta_2 * a1;
 
    Theta2_grad = Theta2_grad + DELTA2;
    Theta1_grad = Theta1_grad + DELTA1;
    
end


Theta1_grad = Theta1_grad./m;

Theta2_grad = Theta2_grad./m;
% plus regularisation


Theta1_grad(:, 2:end) = ((Theta1_grad(:, 2:end))+ (lambda * Theta1(:, 2:end))/m);
%juste
Theta2_grad(:, 2:end) = ((Theta2_grad(:, 2:end))+ (lambda * Theta2(:, 2:end))/m);


  
  



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
