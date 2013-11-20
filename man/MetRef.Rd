\docType{data}

\name{MetRef}

\alias{MetRef}

\title{Nuclear Magnetic Resonance Spectra of Urines}

\description{
   Nuclear magnetic resonance spectra of urines. The data belong to a cohort of 22 healthy donors (11 male and 11 female) where each provided about 40 urine samples over the time course of approximately 2 months, for a total of 873 samples.
}

\usage{data(MetRef)}

\value{
A list with the following elements:
	\item{data}{Gene expression data. A matrix with 873 rows and 450 columns.}
	\item{gender}{Gender index. A vector with 873 elements.}
	\item{donor}{Donor index. A vector with 873 elements.}
  
}

\references{
Assfalg M, Bertini I, Colangiuli D, \emph{et al.} \cr
Evidence of different metabolic phenotypes in humans.\cr
\emph{Proc Natl Acad Sci U S A} 2008;105(5):1420-4.
}
\examples{
data(MetRef)
u=MetRef$data;
u=u[,-which(colSums(u)==0)]
u=scaling(u)$newXtrain
class=as.numeric(as.factor(MetRef$gender))
cc= cmdscale(dist(u))
plot(cc,pch=21,bg=class,xlab="First Component",ylab="Second Component");
}

\keyword{datasets}
