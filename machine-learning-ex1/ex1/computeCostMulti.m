function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
theta_size = size(theta,1);

H = zeros(m,1);
for i = 1:m
    S = 0;
    for j = 1:theta_size
        S = S + theta(j)*X(i,j);
    end
    H(i) = S;
end

Sum = 0;

for i = 1:m
    Sum = Sum + (H(i)-y(i))^2;
end

J = (1/(2*m))*Sum;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
