\name{k.test}
\alias{k.test}
\title{K-Test of Statistical Association}
\description{
This function performs a permutation test using PLS to assess association between the KODAMA output and any additional related parameters such as clinical metadata.
}
\usage{k.test(data, labels, n = 100)}
\arguments{
  \item{data}{a matrix.}
  \item{labels}{a classification vector.}
  \item{n}{number of iterations of the permutation test.}
}

\value{
The p-value of the test.
}
\references{
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.
}

\author{Stefano Cacciatore}

\seealso{\code{\link{KODAMA}}}
\examples{
# data(iris)
# data=iris[,-5]
# labels=iris[,5]
# kk=KODAMA(data,FUN="PLS-DA")
# kkplot=kk$pp
# k1=k.test(kkplot,labels)
# print(k1)
# k2=k.test(kkplot,sample(labels))
# print(k2)
}
\keyword{k.test}
