\name{loads}
\alias{loads}
\title{Variable Ranking}
\description{
This function can be used to extract the variable ranking when KODAMA is performed with the PLS-DA classifier.
}
\usage{
loads(model,method=c("loadings","kruskal.test"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model}{output of KODAMA.}
  \item{method}{method to be used. Choices are "\code{loadings}" and "\code{kruskal.test}".}
}

\value{
The function returns a vector of values indicating the "importance" of each variable. If "\code{method="loadings"} the average of the loading of the first component of PLS models based on the cross-validated accuracy maximized vector is computed. If "\code{method="kruskal.test"} the average of minus logarithm of p-value of Kruskal-Wallis Rank Sum test is computed. 
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
# kk=KODAMA(data,FUN="PLS-DA",f.par=2)
# loads(kk)
}




