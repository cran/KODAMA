\name{transformy}

\alias{transformy}

\title{Conversion Classification Vector to Matrix}

\description{This function converts a classification vector into a classification matrix.}

\usage{
transformy(y)
}

\arguments{
  \item{y}{a vector or factor.}     
}

\details{
	This function converts a classification vector into a classification matrix. 
}


\value{
    A matrix.
  }


\author{Stefano Cacciatore and Leonardo Tenori}


  \examples{
y=rep(1:10,3)
y
transformy(y)
}

\keyword{transformation}
