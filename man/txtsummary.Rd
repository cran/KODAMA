\name{txtsummary}

\alias{txtsummary}

\title{Median and Coefficient Interval}

\description{Summarization of a numeric vector.}

\usage{
txtsummary (x,digits=0,scientific=FALSE)
}

\arguments{
  \item{x}{a numeric vector.}
  \item{digits}{how many significant digits are to be used.}
  \item{scientific}{either a logical specifying whether result should be encoded in scientific format.}

}

\value{
    The function returns the median and the 95\% coefficient interval of numeric vetor.
  }

\author{Stefano Cacciatore}

\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{http://www.pnas.org/content/111/14/5117}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btw705/2667156/KODAMA-an-R-package-for-knowledge-discovery-and}{Link}
}

  \seealso{\code{\link{categorical_table}},\code{\link{continuous_table}}}

\examples{
data(clinical)

txtsummary(clinical[,"BMI"])


}
\keyword{txtsummary}
