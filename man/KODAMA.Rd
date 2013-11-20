\name{KODAMA}

\alias{KODAMA}

\title{Knowledge Discovery by Accuracy Maximization}

\description{
	KODAMA (KnOwledge Discovery by Accuracy MAximization) is an unsupervised and semisupervised learning algorithm that performs feature extraction from noisy and high-dimensional data. Unlike other data mining methods, the peculiarity of KODAMA is that it is driven by an integrated procedure of cross validation of the results.
}
\usage{
KODAMA(data,M=100,Tcycle=20,
       FUN_VAR=function(x){ceiling(ncol(x))},
       FUN_SAM=function(x){ceiling(nrow(x)*0.75)},
       bagging=FALSE,
       FUN=KNN.CV,
       f.par=list(kn=10),
       W=NULL,
       constrain=NULL,
       fix=rep(FALSE,nrow(data)),
       epsilon=0.05,
       shake=FALSE)
}

\arguments{
  \item{data}{a matrix.}
  \item{M}{
  	number of iterative processes (step I-III). 
}
  \item{Tcycle}{
  	number of interative cycle that leads to the maximization of cross-validated accuracy.
}
  \item{FUN_VAR}{
  	function to select the number of variable to select randomly. By default all variable are taken. 
}
  \item{FUN_SAM}{
  	function to select the number of sample to select randomly. By default the 75% of all samples are taken.
}
  \item{bagging}{
  	If it Should sampling be with replacement, \code{bagging = TRUE}. By default \code{bagging = FALSE}
}
  \item{FUN}{
  	classifier to be consider. Choices are "\code{\link{KNN.CV}}", "\code{\link{PLS.SVM.CV}}" , and "\code{\link{PCA.CA.KNN.CV}}".
}
  \item{f.par}{
  	parameters of the classifier. 
}
  \item{W}{
  	a vector of \code{nrow(data)} elements. The KODAMA procedure can be started by different initializations of the vector \code{W}. Without any a priori information the vector \code{W} can be initializated with each element being different from the others (i.e., each sample categorized in a one-element class). Alternatively, the vector \code{W} can be initialized by a clustering procedure, such as \code{\link{kmeans}}.  
}
  \item{constrain}{
  	a vector of \code{nrow(data)} elements. Supervised constraints can be imposed by linking some samples in such a way that if one of them is changed the linked ones must change in the same way (i.e., they are forced to belong to the same class) during the maximization of the cross-validation accuracy procedure. Sample with the same identificative constrain will be forced to stay together.
}
  \item{fix}{
  	a vector of \code{nrow(data)} elements. The values of this vector must to be \code{TRUE} or \code{FALSE}. By default all elements are \code{FALSE}. Samples with the \code{TRUE} fix value will not change the class label defined in \code{W} during the maximization of the cross-validation accuracy procedure. 
}
  \item{epsilon}{
  	cut-off value for low proximity. High proximity are typical of intracluster relationships, whereas low proximities are expected for interluster relationships. Very low proximities between samples are ignored by (default) setting \code{epsilon = 0.05}.
}
  \item{shake}{
  	if \code{shake = FALSE} the cross-validated accuracy is computed with the class defined in \code{W} else the it is not, before the maximization of the cross-validation accuracy procedure.
  }
}

\details{
KODAMA consists of five steps. For a simple description of the method, we can divide KODAMA into two parts: (i) the maximization of cross-validated accuracy by an iterative process (step I and II), resulting in the construction of a proximity matrix (step III), and (ii) the definition of a dissimilarity matrix (step IV and V). The first part entails the core idea of KODAMA, that is, the partitioning of data guided by the maximization of the cross-validated accuracy. At the begininng of this part, a fraction of the total samples (defined by \code{FUN_SAM}) are randomly selected from the original data. The whole iterative process (step I-III) is repeated \code{M} times to average the effects owing to the randomness of the iterative procedure. Each time that this part is repeated, a different fraction of sample is selected. The second part aims at collectioning and processing these results by costructing a dissimilarity matrix to provide a holistic view of the data while maintaining their intrinsic structure (steps IV and V).
}

\value{
    The function returns a list with 4 items:
    \item{dissimilarity}{a dissimilarity matrix.}
    \item{acc}{a vector with the \code{M} cross-validated accuracies.}
    \item{proximity}{a proximity matrix.}
    \item{v}{a matrix containing the all classification obtained maximizing the cross-validation accuracy.}
}

\references{
Cacciatore S, Luchinat C, Tenori L.\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
}

\author{Stefano Cacciatore and Leonardo Tenori}


\seealso{
\code{\link{cmdscale}}
}

\examples{

# data(iris)
# kk=KODAMA(iris[,-5])
# pp = cmdscale(kk$dissimilarity)
# plot(pp,col=rep(2:4,each=50))
#
#
#
# WARNING: The next example is high computational extensive
#
# data(MetRef);
# u=MetRef$data;
# u=u[,-which(colSums(u)==0)]
# u=scaling(u)$newXtrain
# class=as.factor(unlist(MetRef$donor))
# kk=KODAMA(u,FUN=PCA.CA.KNN.CV, W=function(x) as.numeric(kmeans(x,50)$cluster))
# pp = cmdscale(kk$dissimilarity)
# plot(pp,col=class)
# pp = cmdscale(kk$dissimilarity)
# plot(pp,col=class)

}
