\name{classprob}

\alias{classprob}

\title{Determines the Prevalence of Each Class}

\description{This function determines the prevalence or probability for each class of a vector (treated as factor).}

\usage{
classprob(x)
}

\arguments{
  \item{x}{a vector.}     
}

\details{
	This function treats the input vector as a factor and determines the probability for each level (class) of the factor. The order of the returned probabilities is the order of the \code{levels} command, which defaults to numeric of alphabetic order.
}


\value{
    A vector whose length is equal to the number of levels in the input. The order is numerically or alphabetically increasing. Note the factors may have levels which are not present in the vector. See examples for details.
  }

  \author{Atina Dunlap Brooks}

  \seealso{\code{\link{majority}}}

  \examples{
#calculate probabilities
x <- sample(c("a","b","c","d","e"), 10, replace=TRUE)
classprob(x)
#labels the probabilities
levels(as.factor(x))

#to see levels which aren't represented in the vector
x <- as.factor(c("a","a","a","b","b","c"))
levels(x)
#now remove the "c"
x <- x[1:5]
#but "c" is still a level
levels(x)
#and the probability is calculated for it
classprob(x)
}


