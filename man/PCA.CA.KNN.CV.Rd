\name{PCA.CA.KNN.CV}

\alias{PCA.CA.KNN.CV}

\title{Cross-Validation with PCA-CA-kNN.}

\description{This is function performs a 10-fold cross validation on a given data set using k nearest neighbors (kNN) classifier. The kNN classifier is performed on the score of the Principal Component Analysis (PCA) and Canonical Analysis (CA). The output is a vector of predicted labels.}

\usage{
PCA.CA.KNN.CV(x,cl,constrain,kn=10,variance=0.9)
}

\arguments{
  \item{x}{a matrix.}     
  \item{cl}{a classification vector.}
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample with the same identificative constrain will be split in the training set or in the test test of cross-validation together.}
    \item{kn}{the number of nearest neighbors to consider.} 
  \item{variance}{the number of the pricipal component of the PCA is selected on the base of the amount of selected variance. (by default = 0.9). } 

}

\details{
	PCA-CA-kNN classifier was used successfully in \emph{Wallner-Liebmann, et al.} (2012) and \emph{Saccenti, et al.} (2012) to classify metabolomic data.
}


\value{
    The function returns a vector of predicted labels.
  }

\references{
Wallner-Liebmann S, Gralka E, Tenori L \emph{et al.}\cr
The impact of free or standardized lifestyle and urine sampling protocol on metabolome recognition accuracy.\cr
\emph{Genes Nutr} 2015;10:441
\cr
\cr
Saccenti E, Tenori L, Verbruggen P, \emph{et al.}\cr
Of monkeys and men: a metabolomic analysis of static and dynamic urinary metabolic phenotypes in two species. \cr
\emph{PLoS One} 2014;9(9):e106077.

}

\author{Stefano Cacciatore and Leonardo Tenori}

  \seealso{\code{\link{PLS.SVM.CV}},\code{\link{KNN.CV}}}

  \examples{
data(MetRef);
u=MetRef$data;
u=u[,-which(colSums(u)==0)]
u=scaling(u)$newXtrain
class=as.factor(unlist(MetRef$donor))
results=PCA.CA.KNN.CV(u,class,1:length(class))
levels(results)=levels(class)
table(results,class)
}

\keyword{croos-validation}
