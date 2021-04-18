\docType{data}

\name{MetRef}

\alias{MetRef}

\title{Nuclear Magnetic Resonance Spectra of Urine Samples}

\description{The data belong to a cohort of 22 healthy donors (11 male and 11 female) where each provided about 40 urine samples over the time course of approximately 2 months, for a total of 873 samples. Each sample was analysed by Nuclear Magnetic Resonance Spectroscopy. Each spectrum was divided in 450 spectral bins.
}

\usage{data(MetRef)}

\value{
A list with the following elements:
	\item{data}{Metabolomic data. A matrix with 873 rows and 450 columns.}
	\item{gender}{Gender index. A vector with 873 elements.}
	\item{donor}{Donor index. A vector with 873 elements.}

}


\references{
Assfalg M, Bertini I, Colangiuli D, \emph{et al.}	\cr
Evidence of different metabolic phenotypes in humans.\cr
\emph{Proc Natl Acad Sci U S A} 2008;105(5):1420-4. doi: 10.1073/pnas.0705685105.  \href{https://www.pnas.org/content/105/5/1420.long}{Link}
\cr	\cr
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/content/111/14/5117}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btw705/2667156/KODAMA-an-R-package-for-knowledge-discovery-and}{Link}
}

\examples{
\donttest{
data(MetRef)
u=MetRef$data;
u=u[,-which(colSums(u)==0)]
u=normalization(u)$newXtrain
u=scaling(u)$newXtrain
class=as.numeric(as.factor(MetRef$gender))
cc= prcomp(u)$x
plot(cc,pch=21,bg=class,xlab="First Component",ylab="Second Component")

class=as.numeric(as.factor(MetRef$donor))
plot(cc,pch=21,bg=rainbow(22)[class],xlab="First Component",ylab="Second Component")

kk=KODAMA(u)
plot(kk$pp,pch=21,bg=rainbow(22)[class],xlab="First Component",ylab="Second Component")


}
}
\keyword{datasets}
