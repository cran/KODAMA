\name{core}

\alias{core}

\title{Maximization of Cross-Validateed Accuracy Methods}

\description{This function performs the maximization of cross-validated accuracy by an iterative process}

\usage{
core(x, clbest, Tcycle=20, FUN=KNN.CV, f.par=list(kn=10), 
     constrain=NULL, fix=NULL, shake=FALSE)
}

\arguments{
  \item{x}{a matrix.}     
  \item{clbest}{a vector to optimize.}
  \item{Tcycle}{number of interative cycle that leads to the maximization of cross-validated accuracy.}
  \item{FUN}{classifier to be consider. Choices are "\code{\link{KNN.CV}}", "\code{\link{PLS.SVM.CV}}" , and "\code{\link{PCA.CA.KNN.CV}}".} 
  \item{f.par}{parameters of the classifier.} 
  \item{constrain}{
  	a vector of \code{nrow(data)} elements. Supervised constraints can be imposed by linking some samples in such a way that if one of them is changed the linked ones must change in the same way (i.e., they are forced to belong to the same class) during the maximization of the cross-validation accuracy procedure. Sample with the same identificative constrain will be forced to stay together.
  	} 
  \item{fix}{
  	a vector of \code{nrow(data)} elements. The values of this vector must to be \code{TRUE} or \code{FALSE}. By default all elements are \code{FALSE}. Samples with the \code{TRUE} fix value will not change the class label defined in \code{W} during the maximization of the cross-validation accuracy procedure. 
  	} 
  \item{shake}{if \code{shake = FALSE} the cross-validated accuracy is computed with the class defined in \code{W} else the it is not, before the maximization of the cross-validation accuracy procedure.} 
}

\value{
    The function returns a list with 3 items:
    \item{c}{a classification vector with a maximized cross-validated accuracy.}
    \item{a}{the maximum cross-validated accuracy achieved.}
    \item{v}{a vector of all cross-validated accuracy obtained.}
  }

\references{
Cacciatore S, Luchinat C, Tenori L.	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
}
  \author{Stefano Cacciatore and Leonardo Tenori}

  \seealso{\code{\link{KODAMA}}}

  \examples{
# data(iris)
# u=iris[,-5]
# s=sample(1:150,150,TRUE)
# results=core(u,s)
# unique(s)
# unique(results$c)
}

\keyword{maximization}
