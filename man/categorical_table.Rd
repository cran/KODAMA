\name{categorical_table}

\alias{categorical_table}

\title{Categorical Information}

\description{Summarization of the categorical information.}

\usage{
categorical_table (name,cat,label)
}

\arguments{
  \item{name}{the name of the feature.}
  \item{cat}{the information to summarize.}
  \item{label}{the classification of the cohort.}

}

\value{
    The function returns a table with the summarized information. The p-value is computated using the Fisher's test.
  }
\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{http://www.pnas.org/content/111/14/5117}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btw705/2667156/KODAMA-an-R-package-for-knowledge-discovery-and}{Link}
}
\author{Stefano Cacciatore}

  \seealso{\code{\link{continuous_table}}, \code{\link{txtsummary}}}

\examples{
data(clinical)

A=categorical_table("Gender",clinical[,"Gender"],clinical[,"Hospital"])
B=categorical_table("Gleason score",clinical[,"Gleason score"],clinical[,"Hospital"])
C=categorical_table("Ethnicity",clinical[,"Ethnicity"],clinical[,"Hospital"])

D=continuous_table("BMI",clinical[,"BMI"],clinical[,"Hospital"],digits=2)
E=continuous_table("Age",clinical[,"Age"],clinical[,"Hospital"],digits=1)

rbind(A,B,C,D,E)

}
\keyword{categorical_table}
