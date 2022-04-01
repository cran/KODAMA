\name{correlation.test}

\alias{correlation.test}

\title{Continuous Information}

\description{Summarization of the continuous information.}

\usage{
correlation.test (x,
                  y,
                  method = c("pearson", "spearman","MINE"), 
                  name=NA, 
                  perm=100, ...)
}

\arguments{
  \item{x}{a numeric vector.}
  \item{y}{a numeric vector.}
  \item{method}{a character string indicating which correlation method is to be computed. "pearson" (default), "spearman", or "MINE".}
  \item{name}{the name of the feature.}
  \item{perm}{number of permutation needed to estimate the p-value with MINE correlation.}
  \item{...}{further arguments to be passed to or from methods.}
}

\value{
    The function returns a table with the summarized information. 
  }
\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/doi/10.1073/pnas.1220873111}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/33/4/621/2667156}{Link}
}
\author{Stefano Cacciatore}

  \seealso{\code{\link{categorical.test}},\code{\link{continuous.test}}, \code{\link{txtsummary}}}

\examples{
data(clinical)

correlation.test(clinical[,"Age"],clinical[,"BMI"],name="correlation between Age and BMI")

}
\keyword{correlation.test}