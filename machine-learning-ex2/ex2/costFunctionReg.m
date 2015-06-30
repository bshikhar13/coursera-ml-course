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

Sum = 0;
for i = 1:m
    %disp(size(theta));
    %disp('I am here');
    %disp(size(X(i,:)));
    H = sigmoid(theta' *(X(i,:))');
    Sum=Sum + (-1*y(i)*log(H)) - (1-y(i))*log(1-H);
end

theta_sum = 0;

theta_size = size(theta);


for i = 2:theta_size(1)
    theta_sum = theta_sum + theta(i)*theta(i);
end


J = (Sum/m) + (lambda/(2*m))*theta_sum;


for i = 1:theta_size(1)
    Sum = 0;
    for j=1:m
         H = sigmoid(theta' *(X(j,:))');
        Sum = Sum + ((H - y(j))*(X(j,i)));
    end
    grad(i,1) = Sum/m;
end

for i = 2:theta_size(1)
    grad(i,1) = grad(i,1) + (lambda/m)*theta(i,1);
end



% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
