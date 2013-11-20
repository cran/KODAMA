\name{helicoid}

\alias{helicoid}

\title{Helicoid Data Set Generator}

\description{This function creates a data set on the data points are distribuited on a Ulisse Dini's surface.}

\usage{
helicoid(N=1000)
}

\arguments{
  \item{N}{Number of data points.}     
}

\value{
    The function returns a three dimensional data set.
  }
\author{Stefano Cacciatore and Leonardo Tenori}

\references{
Cacciatore S, Luchinat C, Tenori L.	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
}


\seealso{\code{\link{swissroll}},\code{\link{dinisurface}},\code{\link{spirals}}}
  
\examples{
require("rgl")
x=helicoid()
open3d()
plot3d(x, col=rainbow(1050)[-c(1:50)],box=FALSE,type="s",size=1)

}

\keyword{dataset}
