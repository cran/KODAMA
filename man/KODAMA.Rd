\name{KODAMA}

\alias{KODAMA}

\title{Knowledge Discovery by Accuracy Maximization}

\description{
	KODAMA (KnOwledge Discovery by Accuracy MAximization) is an unsupervised and semi-supervised learning algorithm that performs feature extraction from noisy and high-dimensional data. Unlike other data mining methods, KODAMA is driven by an integrated procedure of cross validation of the results.
}
\usage{
KODAMA(data, 
        M = 100, 
        Tcycle = 20, 
        FUN_VAR = function(x) { ceiling(ncol(x)) },
        FUN_SAM = function(x) { ceiling(nrow(x) * 0.75)},
        bagging = FALSE, 
        FUN = c("PLS-DA","KNN"), 
        f.par = 5, 
        W = NULL, 
        constrain = NULL, 
        fix=NULL, 
        epsilon = 0.05,
        dims=2,
        landmarks=5000)
}

\arguments{
  \item{data}{a matrix.}
  \item{M}{
  	number of iterative processes (step I-III).
}
  \item{Tcycle}{
  	number of iterative cycles that leads to the maximization of cross-validated accuracy.
}
  \item{FUN_VAR}{
  	function to select the number of variables to select randomly. By default all variable are taken.
}
  \item{FUN_SAM}{
  	function to select the number of samples to select randomly. By default the 75 per cent of all samples are taken.
}
  \item{bagging}{
  	Should sampling be with replacement, \code{bagging = TRUE}. By default \code{bagging = FALSE}.
}
  \item{FUN}{classifier to be considered. Choices are "\code{PLS-DA}" and "\code{KNN}".
}

  \item{f.par}{
  	parameters of the classifier.
}
  \item{W}{
  	a vector of \code{nrow(data)} elements. The KODAMA procedure can be started by different initializations of the vector \code{W}. Without any \emph{a priori} information the vector \code{W} can be initialized with each element being different from the others (\emph{i.e.}, each sample categorized in a one-element class). Alternatively, the vector \code{W} can be initialized by a clustering procedure, such as \code{\link{kmeans}}.
}
  \item{constrain}{
  	a vector of \code{nrow(data)} elements. Supervised constraints can be imposed by linking some samples in such a way that if one of them is changed the remaining linked samples must change in the same way (\emph{i.e.}, they are forced to belong to the same class) during the maximization of the cross-validation accuracy procedure. Samples with the same identifying constrain will be forced to stay together.
}
  \item{fix}{
  	a vector of \code{nrow(data)} elements. The values of this vector must to be \code{TRUE} or \code{FALSE}. By default all elements are \code{FALSE}. Samples with the \code{TRUE} fix value will not change the class label defined in \code{W} during the maximization of the cross-validation accuracy procedure.
}
  \item{epsilon}{
  	cut-off value for low proximity. High proximity are typical of intracluster relationships, whereas low proximities are expected for intercluster relationships. Very low proximities between samples are ignored by (default) setting \code{epsilon = 0.05}.
}
    \item{dims}{
  dimensions of the configurations of Sammon's non-linear mapping based on the KODAMA dissimilarity matrix.
  }
      \item{landmarks}{
  number of landmarks to use.
  }

}

\details{
KODAMA consists of five steps. These can be in turn divided into two parts: (i) the maximization of cross-validated accuracy by an iterative process (step I and II), resulting in the construction of a proximity matrix (step III), and (ii) the definition of a dissimilarity matrix (step IV and V). The first part entails the core idea of KODAMA, that is, the partitioning of data guided by the maximization of the cross-validated accuracy. At the beginning of this part, a fraction of the total samples (defined by \code{FUN_SAM}) are randomly selected from the original data. The whole iterative process (step I-III) is repeated \code{M} times to average the effects owing to the randomness of the iterative procedure. Each time that this part is repeated, a different fraction of samples is selected. The second part aims at collecting and processing these results by constructing a dissimilarity matrix to provide a holistic view of the data while maintaining their intrinsic structure (steps IV and V). Then, Sammon's non-linear mapping is used to visualise the results of KODAMA dissimilarity matrix. 

}

\value{
    The function returns a list with 4 items:
    \item{dissimilarity}{a dissimilarity matrix.}
    \item{acc}{a vector with the \code{M} cross-validated accuracies.}
    \item{proximity}{a proximity matrix.}
    \item{v}{a matrix containing the all classification obtained maximizing the cross-validation accuracy.}
    \item{pp}{a matrix containing the score of the Sammon's non-linear mapping.}
    \item{res}{a matrix containing all classification vectors obtained through maximizing the cross-validation accuracy.}
    \item{f.par}{parameters of the classifier..}
    \item{entropy}{Shannon's entropy of the KODAMA proximity matrix.}
    \item{landpoints}{indexes of the landmarks used.}
    
}

\author{Stefano Cacciatore and Leonardo Tenori}

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
\donttest{

 data(iris)
 data=iris[,-5]
 labels=iris[,5]
 kk=KODAMA(data,FUN="KNN")
 plot(kk$pp,col=as.numeric(labels), xlab="First component", ylab="Second component",cex=2)

}
}