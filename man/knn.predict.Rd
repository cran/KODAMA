\name{knn.predict}

\alias{knn.predict}

\title{KNN Prediction Routine using Pre-Calculated Distances}

\description{K-Nearest Neighbor prediction method which uses the distances calculated by \code{\link{knn.dist}}.}

\usage{
knn.predict(train, test, y, dist.matrix, k=1,
            agg.meth=if (is.factor(y)) "majority" else "mean",
            ties.meth="min") 
}

\arguments{
  \item{train}{indexes which specify the rows of the \code{dist.matrix} to use as training set.}     
  \item{test}{indexes which specify the rows of the \code{dist.matrix} to use as test set.}     
  \item{y}{a vector of labels.}     
  \item{dist.matrix}{the output from a call to \code{\link{knn.dist}}.}     
  \item{k}{the number of nearest neighbors to consider.}     
  \item{agg.meth}{method to combine responses of the nearest neighbors, defaults to "majority" for classification and "mean" for continuous responses.}  
  \item{ties.meth}{method to handle ties for the k-th neighbor, the default is \code{"min"} which uses all ties, alternatives include \code{"max"} which uses none if there are ties for the k-th nearest neighbor, \code{"random"} which selects among the ties randomly and \code{"first"} which uses the ties in their order in the data.}     


}

\details{
	Predictions are calculated for each test case by aggregating the responses of the k-nearest neighbors among the training cases. \code{k} may be specified to be any positive integer less than the number of training cases, but is generally between 1 and 10.
	The indexes for the training and test cases are in reference to the order of the entire data set as it was passed to \code{\link{knn.dist}}.
	The aggregation may be any named function. By default, classification (factored responses) will use the \code{majority} class function and non-factored responses will use \code{mean}. Other options to consider include \code{min}, \code{max}, and \code{median}.
	The ties are handled using the \code{rank} function. Further information may be found by examining the \code{ties.method} there.
}


\value{
    A vector of predictions whose length is the number of test cases.
  }

\author{Atina Dunlap Brooks}

\seealso{\code{\link{knn.dist}},\code{\link{dist}}}

\examples{
	#a quick classification example
	x1 <- c(rnorm(20, mean=1), rnorm(20, mean=5))
	x2 <- c(rnorm(20, mean=5), rnorm(20, mean=1))
	y=rep(1:2,each=20)
	x <- cbind(x1,x2)
	train <- sample(1:40, 30)
	#plot the training cases
	plot(x1[train], x2[train], col=y[train]+1)
	#predict the other cases
	test <- (1:40)[-train]
	kdist <- knn.dist(x)
	preds <- knn.predict(train, test, y ,kdist, k=3, agg.meth="majority")
	#add the predictions to the plot
	points(x1[test], x2[test], col=as.integer(preds)+1, pch="+")
	#display the confusion matrix
	table(y[test], preds)
}

\keyword{classifier}
