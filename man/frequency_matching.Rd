\name{frequency_matching}

\alias{frequency_matching}

\title{Frequency Matching}

\description{A method to select unbalanced groupd in a cohort.}

\usage{
frequency_matching (data,label,times=5,seed=1234)
}

\arguments{
  \item{data}{a data.frame of data.}
  \item{label}{a classification of the groups.}
  \item{times}{The ratio between the two groups.}
  \item{seed}{a single number for random number generation.}

}



\value{
    The function returns a list with 2 items or 4 items (if a test data set is present):
    \item{data}{the data after the frequency matching.}
    \item{label}{the label after the frequency matching.}
    \item{selection}{the rows selected for the frequency matching.}
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




\examples{
data(clinical)

A=categorical_table("Gender",clinical[,"Gender"],clinical[,"Hospital"])
B=categorical_table("Gleason score",clinical[,"Gleason score"],clinical[,"Hospital"])
C=categorical_table("Ethnicity",clinical[,"Ethnicity"],clinical[,"Hospital"])

D=continuous_table("BMI",clinical[,"BMI"],clinical[,"Hospital"],digits=2)
E=continuous_table("Age",clinical[,"Age"],clinical[,"Hospital"],digits=1)

# Analysis without matching
rbind(A,B,C,D,E)



# The order is important. Right is more important than left in the vector
# So, Ethnicity will be more important than Age
var=c("Age","BMI","Gleason score" ,"Ethnicity")
t=frequency_matching(clinical[,var],clinical[,"Hospital"],times=1)

newdata=clinical[t$selection,]

A=categorical_table("Gender",newdata[,"Gender"],newdata[,"Hospital"])
B=categorical_table("Gleason score",newdata[,"Gleason score"],newdata[,"Hospital"])
C=categorical_table("Ethnicity",newdata[,"Ethnicity"],newdata[,"Hospital"])

D=continuous_table("BMI",newdata[,"BMI"],newdata[,"Hospital"],digits=2)
E=continuous_table("Age",newdata[,"Age"],newdata[,"Hospital"],digits=1)

# Analysis with matching
rbind(A,B,C,D,E)

}
