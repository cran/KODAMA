\name{categorical.test}

\alias{categorical.test}

\title{Categorical Information}

\description{Summarization of the categorical information.}

\usage{
categorical.test (name,x,y,total.column=FALSE,...)
}

\arguments{
  \item{name}{the name of the feature.}
  \item{x}{the information to summarize.}
  \item{y}{the classification of the cohort.}
  \item{total.column}{option to visualize the total (by default = "\code{FALSE}").}
    \item{...}{further arguments to be passed to the function.}

}

\value{
    The function returns a table with the summarized information and The p-value computated using the Fisher's test.
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

  \seealso{\code{\link{correlation.test}},\code{\link{continuous.test}}, \code{\link{txtsummary}}}

\examples{

data(clinical)

hosp=clinical[,"Hospital"]
gender=clinical[,"Gender"]
GS=clinical[,"Gleason score"]
BMI=clinical[,"BMI"]
age=clinical[,"Age"]

A=categorical.test("Gender",gender,hosp)
B=categorical.test("Gleason score",GS,hosp)

C=continuous.test("BMI",BMI,hosp,digits=2)
D=continuous.test("Age",age,hosp,digits=1)

rbind(A,B,C,D)


}
\keyword{categorical.test}
