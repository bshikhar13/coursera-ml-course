function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_size = length(theta);
for iter = 1:num_iters
    J = 0;
    H = zeros(m,1);
    
    for a = 1:m
        Temp = 0;
        for b = 1:theta_size
            Temp = Temp + theta(b,1)*X(a,b);
        end
        H(a,1) = Temp;
    end
    
    Temp = zeros(theta_size,1);
    for b = 1:theta_size
        Sum = 0;
        for a = 1:m
            Sum = Sum+((H(a,1)-y(a,1))*X(a,b));
        end
        Temp(b,1) = Sum;
    end
    
    for i = 1:theta_size
        theta(i) = theta(i) - (alpha/m)*Temp(i,1);
    end

       
    
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
