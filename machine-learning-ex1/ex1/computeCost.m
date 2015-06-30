function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
H = zeros(m,1);
Sum = 0;
for a = 1:m
    H(a,1) = theta(1,1)*X(a,1)+theta(2,1)*X(a,2);
end

for a = 1:m
    Sum = Sum+(H(a,1)-y(a,1))^2;
end
format short g;
J = (1/(2*m))*Sum;

%J=round(J,2);

%   ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end