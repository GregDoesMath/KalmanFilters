clear;
modelFun = @(X, t) X(2) .* exp( -(1 - X(1).*t).^2 ./ (2.*X(3).^2) );
% X = [a, B, varianceSigma]
trueX = [1, 2, 3]';
trueY = @(t) modelFun(trueX, t);

% Generate vector of evenly spaced time values
STEP_SIZE = 0.1;
NUM_MEASUREMENTS = 101;
END_VALUE = (NUM_MEASUREMENTS - 1) * STEP_SIZE;
timeValues = (0 : STEP_SIZE : END_VALUE)';

measurements = generateMeasurements(trueY, 0.2, 0, timeValues);

% Build the Basis function matrix as a function of estimate X=[a,B,sigma]
yPartial_a = @(X, t) ( (X(2) .* t .* (1 - X(1).*t)) ./ X(3).^2 ) .* ...
	exp( -(1 - X(1).*t).^2 ./ (2.*X(3).^2) );
yPartial_B = @(X, t) exp( -(1 - X(1).*t).^2 ./ (2.*X(3).^2) );
yPartial_sigma = @(X, t) ( (X(2) .* (1 - X(1).*t).^2) ./ X(3).^3 ) .* ...
	exp( -(1 - X(1).*t).^2 ./ (2.*X(3).^2) );
basisFunMatrix = @(X) [ yPartial_a(X, timeValues), ...
	yPartial_B(X, timeValues), yPartial_sigma(X, timeValues) ];

basisFun1 = @(X) modelFun(X, timeValues);

% Initialization
estimatesX = cell(100,1); % Initialize cell array of estimates
weightMatrix = eye(101);
TOLERANCE = 10^-5;
estimatesX{1} = [1.2, 1.5, 0.9]' .* trueX; % Initial Guess
stepSizeEta = 10 * norm(basisFunMatrix(estimatesX{1})' * weightMatrix * ...
	basisFunMatrix(estimatesX{1})); % Initial Eta
f = 10;
curlyH = @(X) diag( diag(basisFunMatrix(X)' * weightMatrix * basisFunMatrix(X)) );
basisFun1 = @(X) modelFun(X, timeValues);
deltaYc = @(X) measurements - basisFun1(X);
deltaXc = @(X, eta) inv( (basisFunMatrix(X)' * weightMatrix * basisFunMatrix(X)) ...
	+ eta .* curlyH(X) ) * basisFunMatrix(X)' * weightMatrix * deltaYc(X);
errorVector = @(X, eta) deltaYc(X) - basisFunMatrix(X) * deltaXc(X, eta);
residualErrorJ = @(X, eta) (1/2) * errorVector(X, eta)' * weightMatrix * ...
	errorVector(X, eta);

rssJ = cell(100,1); % Initialize cell array of J
rssJ{1} = residualErrorJ(estimatesX{1}, stepSizeEta);
relativeErrorJ = 10;

iteration = 1;
while relativeErrorJ > TOLERANCE
	currentX = estimatesX{iteration};
	changeInX = deltaXc(currentX, iteration);
	estimatesX{iteration + 1} = currentX + changeInX;
	rssJ{iteration + 1} = residualErrorJ(estimatesX{iteration + 1}, stepSizeEta);
	if rssJ{iteration + 1} >= rssJ{iteration}
		estimatesX{iteration + 1} = estimatesX{iteration};
		stepSizeEta = f .* stepSizeEta;
	else
		stepSizeEta = stepSizeEta ./ f;
	end
	relativeErrorJ = norm( rssJ{iteration + 1} - rssJ{iteration} ) ./ ...
		rssJ{iteration};
	iteration = iteration + 1;
end

% Clean up: delete empty entries for cell arrays and convert to matrices
estimatesMatrix = cell2mat(estimatesX')';
rssJ = cell2mat(rssJ);

sprintf('No. of iterations: %d', iteration)
sprintf('Final estimates:')
sprintf('a = %d \nB = %d \nSigma = %d', ...
	estimatesMatrix(end, 1), estimatesMatrix(end, 2), estimatesMatrix(end, 3))

figure(1)
plot(timeValues, trueY(timeValues), '-k', 'LineWidth', 2);
hold on;
for k = 1: iteration
	estimatesInModel = modelFun(estimatesX{k}, timeValues);
	plot(timeValues, estimatesInModel);
end
hold off;

figure(2)
plot(rssJ);