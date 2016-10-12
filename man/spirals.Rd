\name{spirals}

\alias{spirals}

\title{Spirals Data Set Generator}

\description{Produces a data set of spiral clusters.}

\usage{
spirals(n=c(100,100,100),sd=c(0,0,0))
}



\arguments{
  \item{n}{a vector of integer. The length of the vector is the number of clusters and each number corresponds to the number of data points in each cluster.}
  \item{sd}{amount of noise for each spiral.}

}

\value{
    The function returns a two dimensional data set.
  }
  \references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
\cr
\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.

}
\author{Stefano Cacciatore and Leonardo Tenori}


  \seealso{\code{\link{helicoid}},\code{\link{dinisurface}},\code{\link{swissroll}}}
  \examples{
pdf(file="spirals.pdf")
par(mfrow=c(2,2),mai=c(0,0,0,0))
v1=spirals(c(100,100,100),c(0.1,0.1,0.1))
plot(v1,col=rep(2:4,each=100))
v2=spirals(c(100,100,100),c(0.1,0.2,0.3))
plot(v2,col=rep(2:4,each=100))
v3=spirals(c(100,100,100,100,100),c(0,0,0.2,0,0))
plot(v3,col=rep(2:6,each=100))
v4=spirals(c(20,40,60,80,100),c(0.1,0.1,0.1,0.1,0.1))
plot(v4,col=rep(2:6,c(20,40,60,80,100)))
dev.off()
}

\keyword{dataset}
