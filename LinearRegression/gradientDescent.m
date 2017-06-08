function [theta, J_history, theta_hist, grad_hist] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_hist = zeros(num_iters, 2);
grad_hist = zeros(num_iters, 2);



for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    grad = zeros(2,1);
    for i=1:m
        grad(1,1) = grad(1,1) + (1/m)*(X(i,:)*theta-y(i));
        grad(2,1) = grad(2,1) + (1/m)*(X(i,:)*theta-y(i))*X(i,2);
    end
    %computeCost(X, y, theta)
    
    theta = theta-alpha*grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    theta_hist(iter,:) = theta';
    grad_hist(iter,:) = grad;
end

end
