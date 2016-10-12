\docType{data}

\name{USA}

\alias{USA}

\title{State of the Union Data Set}

\description{
This dataset consists of the spoken, not written, addresses from 1900 until the sixth address by Barack Obama in 2014. Punctuation characters, numbers, words shorter than three characters, and stop-words (e.g., "that", "and", and "which") were removed from the dataset. This resulted in a dataset of 86 speeches containing 834 different meaningful words each. Term frequency-inverse document frequency (TF-IDF) was used to obtain feature vectors. It is often used as a weighting factor in information retrieval and text mining. The TF-IDF value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus, which helps to control for the fact that some words are generally more common than others.
}

\usage{data(USA)}

\value{
A list with the following elements:
	\item{data}{TF-IDF data. A matrix with 86 rows and 834 columns.}
	\item{year}{Year index. A vector with 86 elements.}
	\item{president}{President index. A vector with 86 elements.}
}


  \references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
\cr
\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.
}
\examples{
# Here is reported the analysis on the State of the Union
# of USA president as shown in Cacciatore, et al. (2014)
#
# data(USA)
# kk=KODAMA(USA$data)
# cc=cmdscale(kk$dissimilarity)
# par(cex=0.5,mar=c(15,6,2,2));
# plot(USA$year,cc[,1],axes=F,pch=20,xlab="",ylab="First Component");
# axis(1,at=USA$year,labels=rownames(USA$data),las=2);
# axis(2,las=2);
# box()
#

}

\keyword{datasets}
