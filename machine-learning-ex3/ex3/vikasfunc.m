function g = vikasfunc(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

%g = 1.0 ./ (1.0 + exp(-z));
g = sin(z) .* cos(z);
end