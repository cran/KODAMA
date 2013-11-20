\name{knn.dist}

\alias{knn.dist}

\title{Calculates the Distances for KNN Predictions}

\description{
	The distances to be used for K-Nearest Neighbor (KNN) predictions are calculated and returned as a symmetric matrix. Distances are calculated by \code{\link{dist}}.
}

\usage{
knn.dist(x, dist.meth = "euclidean", p = 2)
}

\arguments{
  \item{x}{a matrix of data.}     
  \item{dist.meth}{the distance to be used in calculating the neighbors. Any method valid in function \code{\link{dist}} is valid.}
  \item{p}{the power of the Minkowski distance.}
}

\details{
	This function calculates the distances to be used by \code{\link{knn.predict}}. Distances are calculated between all cases. In the traditional scenario. The advantage to calculating distances in a separate step prior to prediction, is that these claculations only need to be performed once. So, for example, cross-validation to select \emph{k} can be performed on many values of \emph{k}, with different cross-validation splits, all using a single run of \code{knn.dist}.
	The default method for calculating distances is the \code{"euclidean"} distance, which is the method used by the \code{knn} function from the \code{class} package. Alternative methods may be used here. Any method valid for the function \code{\link{dist}} is valid here. The parameter \code{p} may be specified with the Minkowski distance to use the \code{p} norm as the distance method.
  
}


\value{
    a square symmetric matrix whose dimensions are the number of rows in the original data. The diagonal contains zeros, the off diagonal entries will be >=0..
  }

  \author{Atina Dunlap Brooks}

  \seealso{\code{\link{knn.predict}},\code{\link{dist}}}

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

\keyword{distance}
