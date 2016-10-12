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


  \examples{
y=rep(1:10,3)
print(y)
z=transformy(y)
print(z)
}

\keyword{transformation}
