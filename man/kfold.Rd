\name{kfold}

\alias{kfold}

\title{k-Fold Partitioning}

\description{This function computes the k-fold partitioning of a vector. Each record in a vector is randomly assigned to a group. Group numbers are between 1 and k.}

\usage{
kfold(constrain,k=10)
}

\arguments{
  \item{constrain}{a vector of labels.}     
  \item{k}{number of groups.}
}

\value{
    A vector with group assignmnets.
}

\author{Stefano Cacciatore and Leonardo Tenori}

\examples{
kfold(1:100, k=10)  	
kfold(rep(1:20,each=5), k=10)
}

\keyword{partitioning}
