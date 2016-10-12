\name{pls.kodama}

\alias{pls.kodama}

\title{Partial Least Squares regression.}

\description{Partial Least Squares (PLS) regression for test set from training set.}

\usage{
pls.kodama(Xtrain, Ytrain, Xtest, ncomp) 

}

\arguments{
  \item{Xtrain}{a matrix of training set cases.}
  \item{Ytrain}{a classification vector.}
  \item{Xtest}{a matrix of test set cases.}
  \item{ncomp}{the number of components to consider.}

}

\value{
A list with the following components:
  \item{B}{the (p x m x length(ncomp)) matrix containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Ypred}{the (ntest x m x length(ncomp)) containing the predicted values of the response variables for the observations from Xtest. The third dimension of the matrix Ypred corresponds to the number of PLS components used to compute the regression coefficients.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{T}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}
  \item{meanX}{the p-vector containing the means of the columns of Xtrain.}

  }

  \references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22.
\cr
\cr
Cacciatore S, Tenori L, Luchinat C, Bennet PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics}. Submitted.
}

\author{Stefano Cacciatore}

\seealso{\code{\link{KODAMA}}}

  \examples{
data(iris)
data=iris[,-5]
labels=iris[,5]
ss=sample(150,15)
ncomponent=3
z=pls.kodama(data[-ss,], labels[-ss], data[ss,], ncomp=ncomponent) 
table(z$Ypred[,ncomponent],labels[ss])
}

