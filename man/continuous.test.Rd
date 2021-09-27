\name{continuous.test}

\alias{continuous.test}

\title{Continuous Information}

\description{Summarization of the continuous information.}

\usage{
continuous.test (name,
                 x,    
                 y,
                 digits = 3,
                 scientific = FALSE, 
                 range = c("IQR","95\%CI"), 
                 logchange = TRUE,
                 pos=1, ...) 
}

\arguments{
  \item{name}{the name of the feature.}
  \item{x}{the information to summarize.}
  \item{y}{the classification of the cohort.}
  \item{digits}{how many significant digits are to be used.}
  \item{scientific}{either a logical specifying whether result should be encoded in scientific format.}
  \item{range}{the range to be visualized.}
  \item{logchange}{either a logical specifying whether log2 of fold change  should be visualized.}
  \item{pos}{a value indicating the position of range to be visualized. 1 for column, 2 for row.}
  \item{...}{further arguments to be passed to or from methods.}
}

\value{
    The function returns a list containg a table with the summarized information and the relative p-value. If the number of group is equal to two, the p-value is computed using the Wilcoxon rank-sum test, Kruskal-Wallis test otherwise.
  }
\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/content/111/14/5117}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btw705/2667156/KODAMA-an-R-package-for-knowledge-discovery-and}{Link}
}
\author{Stefano Cacciatore}

  \seealso{\code{\link{correlation.test}}, \code{\link{categorical.test}}, \code{\link{txtsummary}}}

\examples{
data(clinical)
A=categorical.test("Gender",clinical[,"Gender"],clinical[,"Hospital"])
B=categorical.test("Gleason score",clinical[,"Gleason score"],clinical[,"Hospital"])

C=continuous.test("BMI",clinical[,"BMI"],clinical[,"Hospital"],digits=2,pos=2)
D=continuous.test("Age",clinical[,"Age"],clinical[,"Hospital"],digits=1,pos=2)

rbind(A,B,C,D)


}
\keyword{continuous.test}