\name{knn.double.cv}

\alias{knn.double.cv}

\title{Cross-Validation with k-Nearest Neighbors algorithm.}

\description{This function performs a 10-fold cross validation on a given data set using \emph{k}-Nearest Neighbors (\emph{k}NN) model. To assess the prediction ability of the model, a 10-fold cross-validation is conducted by generating splits with a ratio 1:9 of the data set, that is by removing 10\% of samples prior to any step of the statistical analysis, including PLS component selection and scaling. Best number of component for PLS was carried out by means of 10-fold cross-validation on the remaining 90\% selecting the best Q2y value. Permutation testing was undertaken to estimate the classification/regression performance of predictors.}

\usage{
knn.double.cv(Xdata,
              Ydata,
              constrain=1:nrow(Xdata),
              compmax=min(5,c(ncol(Xdata),nrow(Xdata))),
              perm.test=FALSE,
              optim=TRUE,
              scaling = c("centering","autoscaling"),
              times=100,
              runn=10)
}

\arguments{
  \item{Xdata}{a matrix.}
  \item{Ydata}{the responses. If Ydata is a numeric vector, a regression analysis will be performed. If Ydata is factor, a classification analysis will be performed. }
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample with the same identifying constrain will be split in the training set or in the test set of cross-validation together.}
  \item{compmax}{the number of k to be used for classification.}
  \item{perm.test}{a classification vector.}
  \item{optim}{if perform the optmization of the number of k.}
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}" or "\code{autoscaling}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{times}{number of cross-validations with permutated samples}
    \item{runn}{number of cross-validations loops.}

}

\value{
A list with the following components:
  \item{Ypred}{the vector containing the predicted values of the response variables obtained by cross-validation.}
  \item{Yfit}{the vector containing the fitted values of the response variables.}
  \item{Q2Y}{Q2y value.}
  \item{R2Y}{R2y value.}
  \item{conf}{The confusion matrix (only in classification mode).}
  \item{acc}{The cross-validated accuracy (only in classification mode).}

  \item{txtQ2Y}{a summary of the Q2y values.}
  \item{txtR2Y}{a summary of the R2y values.}
  }


\author{Stefano Cacciatore}


\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/doi/10.1073/pnas.1220873111}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/33/4/621/2667156}{Link}
}
\examples{
\donttest{
 data(iris)
 data=iris[,-5]
 labels=iris[,5]
 pp=knn.double.cv(data,labels)
 print(pp$Q2Y)
 table(pp$Ypred,labels)
 

 data(MetRef)
 u=MetRef$data;
 u=u[,-which(colSums(u)==0)]
 u=normalization(u)$newXtrain
 u=scaling(u)$newXtrain
 pp=knn.double.cv(u,as.factor(MetRef$donor))
 print(pp$Q2Y)
 table(pp$Ypred,MetRef$donor)

}
}
