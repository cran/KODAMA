\name{multi_analysis}

\alias{multi_analysis}

\title{Continuous Information}

\description{Summarization of the continuous information.}

\usage{
multi_analysis  (data, 
                 y, 
                 FUN=c("continuous.test","correlation.test"), ...)                 
}

\arguments{
  \item{data}{the matrix containing the continuous values. Each row corresponds to a different sample. Each column corresponds to a different variable.}
  \item{y}{the classification of the cohort.}
  \item{FUN}{function to be considered. Choices are "\code{continuous.test}" and "\code{correlation.test}"}
  \item{...}{further arguments to be passed to or from methods.}
}

\value{
    The function returns a table with the summarized information. If the number of group is equal to two, the p-value is computed using the Wilcoxon rank-sum test, Kruskal-Wallis test otherwise.
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

  \seealso{\code{\link{categorical.test}},\code{\link{continuous.test}},\code{\link{correlation.test}}, \code{\link{txtsummary}}}

\examples{
data(clinical)


multi_analysis(clinical[,c("BMI","Age")],clinical[,"Hospital"],FUN="continuous.test")

}
\keyword{multi_analysis}