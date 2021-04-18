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
\value{
No return value.
}
\author{Stefano Cacciatore}

\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/content/111/14/5117}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btw705/2667156/KODAMA-an-R-package-for-knowledge-discovery-and}{Link}
}


\seealso{\code{\link{KODAMA}}}
\examples{
\donttest{
 data=as.matrix(iris[,-5])
 kk=KODAMA(data)
 mcplot(kk)
}
}