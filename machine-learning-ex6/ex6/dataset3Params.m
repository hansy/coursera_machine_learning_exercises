function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_values     = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
length_C     = length(C_values);
length_sigma = length(sigma_values);

best_index_C = 5;
best_index_sigma = 4;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


init_predict = svmPredict(svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)), Xval);

best_error = mean(double(init_predict ~= yval));

for i = 1:length_C
  for j = 1:length_sigma
    C_test      = C_values(i);
    sigma_test  = sigma_values(j);
    model       = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    err         = mean(double(predictions ~= yval));

    if (err < best_error)
      best_error       = err;
      best_index_C     = i;
      best_index_sigma = j;
    end 
  end
end

C     = C_values(best_index_C);
sigma = sigma_values(best_index_sigma);







% =========================================================================

end
