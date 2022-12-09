\name{knn.kodama}

\alias{knn.kodama}

\title{k-Nearest Neighbors Classifier.}

\description{k-nearest neighbour classification for a test set from a training set.}

\usage{
knn.kodama(Xtrain, 
           Ytrain, 
           Xtest,
           Ytest=NULL, 
           k, 
           scaling = c("centering","autoscaling"),
           perm.test=FALSE,
           times=1000)

}

\arguments{
  \item{Xtrain}{a matrix of training set cases.}
  \item{Ytrain}{a classification vector.}
  \item{Xtest}{a matrix of test set cases.}
  \item{Ytest}{a classification vector.}
  \item{k}{the number of nearest neighbors to consider.}
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}" or "\code{autoscaling}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{perm.test}{a classification vector.}
  \item{times}{a classification vector.}

}

\details{
The function utilizes the Approximate Nearest Neighbor (ANN) C++ library, 
which can give the exact nearest neighbours or (as the name suggests) 
approximate nearest neighbours to within a specified error bound.  For more 
information on the ANN library please visit http://www.cs.umd.edu/~mount/ANN/.
}


\value{The function returns a vector of predicted labels.}

\author{Stefano Cacciatore and Leonardo Tenori}


  \references{
Bentley JL (1975)\cr
Multidimensional binary search trees used for associative search. \cr
\emph{Communication ACM} 1975;18:309-517.\cr\cr


Arya S, Mount DM\cr
Approximate nearest neighbor searching\cr
\emph{Proc. 4th Ann. ACM-SIAM Symposium on Discrete Algorithms (SODA'93)};271-280.\cr\cr
  
Arya S, Mount DM, Netanyahu NS, Silverman R, Wu AY\cr
An optimal algorithm for approximate nearest neighbor searching\cr
\emph{Journal of the ACM} 1998;45:891-923.\cr\cr



Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/doi/10.1073/pnas.1220873111}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/33/4/621/2667156}{Link}
}

\seealso{\code{\link{KODAMA.matrix}},\code{\link{KODAMA.visualization}}}

\examples{
 data(iris)
 data=iris[,-5]
 labels=iris[,5]
 ss=sample(150,15)

 z=knn.kodama(data[-ss,], labels[-ss], data[ss,], k=5) 
 table(z$Ypred[,5],labels[ss])
}
\keyword{cross-validation}
