\name{core_cpp}

\alias{core_cpp}

\title{Maximization of Cross-Validateed Accuracy Methods}

\description{This function performs the maximization of cross-validated accuracy by an iterative process}

\usage{
core_cpp(x, 
         xTdata=NULL,
         clbest, 
         Tcycle=20, 
         FUN=c("KNN","PLS-DA"), 
         fpar=2, 
         constrain=NULL, 
         fix=NULL, 
         shake=FALSE)
}

\arguments{
  \item{x}{a matrix.}
  \item{xTdata}{a matrix for projections. This matrix contains samples that are not used for the maximization of the cross-validated accuracy. Their classification is obtained by predicting samples on the basis of the final classification vector.}
  \item{clbest}{a vector to optimize.}
  \item{Tcycle}{number of iterative cycles that leads to the maximization of cross-validated accuracy.}
  \item{FUN}{classifier to be consider. Choices are "\code{KNN}" and "\code{PLS-DA}".}
  \item{fpar}{parameters of the classifier. If the classifier is \code{KNN}, \code{fpar} represents the number of neighbours. If the classifier is \code{PLS-DA}, \code{fpar} represents the number of components.}
  \item{constrain}{
  	a vector of \code{nrow(data)} elements. Supervised constraints can be imposed by linking some samples in such a way that if one of them is changed, all other linked samples change in the same way (\emph{i.e.}, they are forced to belong to the same class) during the maximization of the cross-validation accuracy procedure. Samples with the same identifying constrain will be forced to stay together.
  	}
  \item{fix}{
  	a vector of \code{nrow(data)} elements. The values of this vector must  be \code{TRUE} or \code{FALSE}. By default all elements are \code{FALSE}. Samples with the \code{TRUE} fix value will not change the class label defined in \code{W} during the maximization of the cross-validation accuracy procedure. For more information refer to Cacciatore, \emph{et al.} (2014).
  	}
  \item{shake}{if \code{shake = FALSE} the cross-validated accuracy is computed with the class defined in \code{W}, before the maximization of the cross-validation accuracy procedure.}
}

\value{
    The function returns a list with 3 items:
    \item{clbest}{a classification vector with a maximized cross-validated accuracy.}
    \item{accbest}{the maximum cross-validated accuracy achieved.}
    \item{vect_acc}{a vector of all cross-validated accuracies obtained.}
    \item{vect_proj}{a prediction of samples in \code{xTdata} matrix using the vector clbest. This output is present only if \code{xTdata} is not \code{NULL}.}
    
  }

\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.	\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.
}
  \author{Stefano Cacciatore and Leonardo Tenori}

  \seealso{\code{\link{KODAMA}}}

  \examples{
# Here, the famous (Fisher's or Anderson's) iris data set was loaded
data(iris)
u=as.matrix(iris[,-5])
s=sample(1:150,150,TRUE)

# The maximization of the accuracy of the vector s is performed
results=core_cpp(u, clbest=s,fpar = 2)

print(as.numeric(results$clbest))

}

