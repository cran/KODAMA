\name{PLS.SVM.CV}

\alias{PLS.SVM.CV}

\title{Cross-Validation with Support Vector Machine.}

\description{This is function performs a 10-fold cross validation on a given data set using the Support Vector Machine (SVM) classifier. The SVM classifier is performed on the score of the Partial least squares (PLS). The output is a vector of predicted labels. }

\usage{
PLS.SVM.CV(x,cl,constrain,ncomp=5, ...)
}

\arguments{
  \item{x}{a matrix.}     
  \item{cl}{a classification vector.}
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample with the same identificative constrain will be split in the training set or in the test test of cross-validation together.}
  \item{ncomp}{number of component of PLS to consider.} 
  \item{...}{parameter for \code{\link{svm}} classifier.} 
}

\details{
	PLS-SVM classifier was used successfully in \emph{Bertini, et al.} (2012) and \emph{Aimetti, et al.} (2012) to classify metabolomic data.
}


\value{
    The function returns a vector of predicted labels.
  }

  \references{
Bertini I, Cacciatore S, Jensen BV, \emph{et al.}\cr
Metabolomic NMR fingerprinting to identify and predict survival of patients with metastatic colorectal cancer.\cr
\emph{Cancer Res} 2012;72(1):356-64.
\cr
\cr
Aimetti M, Cacciatore S, Graziano A, Tenori L.\cr
Metabonomic analysis of saliva reveals generalized chronic periodontitis signature.\cr
\emph{Metabolomics} 2012;8(3):465-74.

}

\author{Stefano Cacciatore and Leonardo Tenori}

  \seealso{\code{\link{KNN.CV}},\code{\link{PCA.CA.KNN.CV}}}

  \examples{
data(lymphoma)
class=as.factor(unlist(lymphoma$class))
results=PLS.SVM.CV(lymphoma$data,class,1:length(class))
levels(results)=levels(class)
table(results,class)
}

\keyword{croos-validation}
