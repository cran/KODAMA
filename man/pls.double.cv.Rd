\name{pls.double.cv}

\alias{pls.double.cv}

\title{Cross-Validation with PLS-DA.}

\description{This function performs a 10-fold cross validation on a given data set using Partial Least Squares (PLS) model. To assess the prediction ability of the model, a 10-fold cross-validation is conducted by generating splits with a ratio 1:9 of the data set, that is by removing 10\% of samples prior to any step of the statistical analysis, including PLS component selection and scaling. Best number of component for PLS was carried out by means of 10-fold cross-validation on the remaining 90\% selecting the best Q2y value. Permutation testing was undertaken to estimate the classification/regression performance of predictors.}

\usage{
pls.double.cv(Xdata,
              Ydata,
              constrain=1:nrow(Xdata),
              compmax=min(c(ncol(Xdata),nrow(Xdata))),
              perm.test=FALSE)
}

\arguments{
  \item{Xdata}{a matrix.}
  \item{Ydata}{the responses. If Ydata is a numeric vector, a regression analysis will be performed. If Ydata is factor, a classification analysis will be performed. }
  \item{constrain}{a vector of \code{nrow(data)} elements. Sample with the same identifying constrain will be split in the training set or in the test set of cross-validation together.}
  \item{compmax}{the number of latent components to be used for classification.}
  \item{perm.test}{a classification vector.}

}

\value{
A list with the following components:
  \item{B}{the (p x m x length(ncomp)) array containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Ypred}{the vector containing the predicted values of the response variables obtained by cross-validation.}
  \item{Yfit}{the vector containing the fitted values of the response variables.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{T}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}
  \item{Q2Y}{Q2y value.}
  \item{R2Y}{R2y value.}
  \item{R2X}{vector containg the explained variance of X by each PLS component.}

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

\author{Stefano Cacciatore}



\examples{
data(iris)
data=iris[,-5]
labels=iris[,5]
pp=pls.double.cv(data,labels)
print(pp$Q2Y)
table(pp$Ypred,labels)

#
# data(MetRef)
# u=MetRef$data;
# u=u[,-which(colSums(u)==0)]
# u=normalization(u)$newXtrain
# u=scaling(u)$newXtrain
# pp=pls.double.cv(u,as.factor(MetRef$donor))
# print(pp$Q2Y)
# table(pp$Ypred,MetRef$donor)
#
}
