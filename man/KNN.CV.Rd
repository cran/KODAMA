\name{KNN.CV}

\alias{KNN.CV}

\title{Cross-Validation with k-Nearest Neighbors Classifier.}

\description{This is function performs a 10-fold cross validation on a given data set using k nearest neighbors (kNN) classifier. The output is a vector of predicted labels.}

\usage{
KNN.CV(x,cl,constrain,kn=10)
}

\arguments{
  \item{x}{a matrix.}     
  \item{cl}{a classification vector.}
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample with the same identificative constrain will be split in the training set or in the test test of cross-validation together.}
  \item{kn}{the number of nearest neighbors to consider.} 

}

\details{
	Details are described in Cover, \emph{et al.} (1967).
}


\value{
    The function returns a vector of predicted labels.
  }

  \references{
Cover TM, Hart PE.\cr
TNearest neighbor pattern classification.\cr
\emph{IEEE Trans Inf Theory} 1967;13(1):21-7.
}

\author{Stefano Cacciatore and Leonardo Tenori}

\seealso{\code{\link{PLS.SVM.CV}},\code{\link{PCA.CA.KNN.CV}}}

\examples{
data(iris)
u=iris[,-5]
class=as.factor(iris[,5])
results=KNN.CV(u,class,1:length(class))
levels(results)=levels(class)
table(results,class)
}

\keyword{croos-validation}
