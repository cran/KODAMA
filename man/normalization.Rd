\name{normalization}

\alias{normalization}

\title{Normalization methods}

\description{Collection of Different Normalization Methods.}

\usage{
normalization(Xtrain,Xtest=NULL, method = "pqn",ref=NULL)
}

\arguments{
  \item{Xtrain}{a matrix of data (training data set).}     
  \item{Xtest}{a matrix of data (test data set).(by default = NULL).}
  \item{method}{the normalization method to be used. Choices are "\code{none}", "\code{pqn}", "\code{sum}", "\code{median}", "\code{sqrt}" (by default = "\code{pqn}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{ref}{Reference sample for Probabilistic Quotient Normalization. (by default = NULL).} 

}

\details{
	A number of different normalization methods are provided:
  \itemize{
    \item{ "\code{none}": no normalization method is applied.}
    \item{ "\code{pqn}": the Probabilistic Quotient Normalization is computed as described in \emph{Dieterle, et al.} (2006).} 
    \item{ "\code{sum}": samples are normalized to the sum of the absolute value of all variables for a given sample.}
    \item{ "\code{median}": samples are normalized to the median value of all variables for a given sample.}  
    \item{ "\code{sqrt}": samples are normalized to the root of the sum of the squared value value of all variables for a given sample.}  
}
}


\value{
    The function returns a list with 2 items or 4 items (if a test data set is present):
    \item{newXtrain}{a normalized matrix (training data set).}
    \item{coeXtrain}{a vector of normalization coefficient of the training data set.}
    \item{newXtest}{a normalized matrix (test data set).}
    \item{coeXtest}{a vector of normalization coefficient of the test data set.}
  }

  \references{
  	Dieterle F,Ross A, Schlotterbeck G, Senn H.\cr
Probabilistic Quotient Normalization as Robust Method to Account for Diluition of Complex Biological Mixtures. Application in 1H NMR Metabolomics.\cr
\emph{Anal Chem} 2006;78:4281-90.
}

\author{Stefano Cacciatore and Leonardo Tenori}

  \seealso{\code{\link{scaling}}}


\keyword{normalization}
