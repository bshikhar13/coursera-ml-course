function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_size = length(theta);

for iter = 1:num_iters
    J = 0;
    H = zeros(m,1);
    
    for a = 1:m
        H(a,1) = theta(1,1)*X(a,1)+theta(2,1)*X(a,2);   
    end
    Temp = zeros(theta_size,1);
    for b = 1:theta_size
        Sum = 0;
        for a = 1:m
            Sum = Sum+((H(a,1)-y(a,1))*X(a,b));
        end
        Temp(b,1) = Sum;
    end
    
    theta(1) = theta(1) - (alpha/m)*Temp(1,1);
    theta(2) = theta(2) - (alpha/m)*Temp(2,1);
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
