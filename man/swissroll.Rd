\name{swissroll}

\alias{swissroll}

\title{Swiss Roll Data Set Generator}

\description{Computes the Swiss Roll data set of a given number of data points.}

\usage{
swissroll(N=1000)
}

\arguments{
  \item{N}{Number of data points.}     
}

\value{
    The function returns a three dimensional matrix.
  }

\author{Stefano Cacciatore and Leonardo Tenori}

  \references{
Cacciatore S, Luchinat C, Tenori L.	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
\cr
\cr
Balasubramanian M, Schwartz EL.\cr
The isomap algorithm and topological stability. \cr
\emph{Science} 2002;295(5552):7. 
\cr
\cr
Roweis ST, Saul LK.\cr
Nonlinear dimensionality reduction by locally linear embedding.\cr
\emph{Science} 2000;290(5500):2323-6.  

}


  \seealso{\code{\link{helicoid}},\code{\link{dinisurface}},\code{\link{spirals}}}

  \examples{
require("rgl")
x=swissroll()
open3d()
plot3d(x, col=rainbow(1050)[-c(1:50)],box=FALSE,type="s",size=1)

}

\keyword{dataset}
