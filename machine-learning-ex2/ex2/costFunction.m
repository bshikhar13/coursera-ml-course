function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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

J = Sum/m;

theta_size = size(theta);

for i = 1:theta_size(1);
    Sum = 0;
    for j=1:m
         H = sigmoid(theta' *(X(j,:))');
        Sum = Sum + ((H - y(j))*(X(j,i)));
    end
    grad(i,1) = Sum/m;
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
