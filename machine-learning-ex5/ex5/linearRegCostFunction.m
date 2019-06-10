function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X = R(12x2) ; theta = R(2x1) ; y = R(12x1)

hypothesis = X * theta;

J = sum((hypothesis - y) .^ 2) / (2 * m);
JReg = lambda * sum(theta(2:end,1) .^ 2) / (2 * m);

J = J + JReg;

%gradient descent -> theta = theta - alpha * derivative(J(theta))/derivative(theta);
for j = 1:size(grad)
  grad(j) = sum((hypothesis - y) .* X(:,j)) / m;
endfor
%grad(1) = sum((hypothesis - y) .* X(1,2)) / m;
%grad(2) = sum((hypothesis - y) .* X(2,2)) / m;


for i = 2:size(grad)
  grad(i) = grad(i) + ((lambda / m) * theta(i));
endfor
%grad(2) = grad(2) + (lambda / m) * theta(2);





% =========================================================================

grad = grad(:);

end
