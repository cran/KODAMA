\name{mcplot}
\alias{mcplot}
\title{Evaluation of the Monte Carlo accuracy results}
\description{
This function can be used to plot the accuracy values obtained during KODAMA procedure.
}
\usage{
mcplot(model)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model}{output of KODAMA.}

}

  \references{
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.
}

\author{Stefano Cacciatore}

\seealso{\code{\link{KODAMA}}}

\examples{
data=as.matrix(iris[,-5])
kk=KODAMA(data)
mcplot(kk)
}

