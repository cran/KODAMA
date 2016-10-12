\name{scaling}

\alias{scaling}

\title{Scaling methods}

\description{Collection of Different Scaling Methods.}

\usage{
scaling(Xtrain,Xtest=NULL, method = "autoscaling")
}

\arguments{
  \item{Xtrain}{a matrix of data (training data set).}
  \item{Xtest}{a matrix of data (test data set).(by default = NULL).}
  \item{method}{the scaling method to be used. Choices are "\code{none}", "\code{centering}", "\code{autoscaling}", "\code{rangescaling}", "\code{paretoscaling}" (by default = "\code{autoscaling}"). A partial string sufficient to uniquely identify the choice is permitted.}
}

\details{
	A number of different scaling methods are provided:
  \itemize{
    \item "\code{none}": no scaling method is applied.
    \item "\code{centering}": centers the mean to zero.
    \item "\code{autoscaling}": centers the mean to zero and scales data by dividing each variable by the variance.
    \item "\code{rangescaling}": centers the mean to zero and scales data by dividing each variable by the difference between the minimum and the maximum value.
    \item "\code{paretoscaling}": centers the mean to zero and scales data by dividing each variable by the square root of the standard deviation. Unit scaling divides each variable by the standard deviation so that each variance equal to 1.
  }
}


\value{
    The function returns a list with 1 item or 2 items (if a test data set is present):
    \item{newXtrain}{a scaled matrix (training data set).}
    \item{newXtest}{a scale matrix (test data set).}
  }

\author{Stefano Cacciatore and Leonardo Tenori}

  \references{
van den Berg RA, Hoefsloot HCJ, Westerhuis JA, \emph{et al.}\cr
Centering, scaling, and transformations: improving the biological information content of metabolomics data. \cr
\emph{BMC Genomics} 2006;7(1):142.
\cr
\cr
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
\cr
\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.

}
  \seealso{\code{\link{normalization}}}

\examples{
data(MetRef)
u=MetRef$data;
u=u[,-which(colSums(u)==0)]
u=normalization(u)$newXtrain
u=scaling(u)$newXtrain
class=as.numeric(as.factor(MetRef$gender))
cc= prcomp(u)$x
plot(cc,pch=21,bg=class,xlab="First Component",ylab="Second Component")
}

\keyword{scaling}
