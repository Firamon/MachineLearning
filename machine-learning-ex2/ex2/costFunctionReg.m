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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


m = size(X,1);
hypothesis = sigmoid(X * theta);
hypothesisZero = sigmoid(X(1) * theta);
hypothesisMore = sigmoid(X(2:size(X)) * theta);

J = ((1 / m) * sum((-y .* log(hypothesis)) - ((1 - y) .* log(1 - hypothesis)))) + ((lambda / (2*m)) * sum(theta(2:size(theta)) .^ 2));   %(2:size(theta))
grad(1) = (1 / m) * sum((hypothesisZero - y(1)) .* X(1));
grad(2:size(grad)) = ((1 / m) * sum((hypothesisMore - y(2:size(y))) .* X(2:size(X)))) + ((lambda / m) .* theta(2:size(theta)));


% =============================================================

end
