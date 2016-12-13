\name{pls.kodama}

\alias{pls.kodama}

\title{Partial Least Squares regression.}

\description{Partial Least Squares (PLS) regression for test set from training set.}

\usage{
pls.kodama(Xtrain, 
           Ytrain, 
           Xtest, 
           Ytest = NULL, 
           ncomp, 
           scaling = c("centering","autoscaling"),
           perm.test=FALSE,
           times=100) 

}

\arguments{
  \item{Xtrain}{a matrix of training set cases.}
  \item{Ytrain}{a classification vector.}
  \item{Xtest}{a matrix of test set cases.}
  \item{Ytest}{a classification vector.}
  \item{ncomp}{the number of components to consider.}
  \item{scaling}{the scaling method to be used. Choices are "\code{centering}" or "\code{autoscaling}" (by default = "\code{centering}"). A partial string sufficient to uniquely identify the choice is permitted.}
  \item{perm.test}{a classification vector.}
  \item{times}{a classification vector.}

}

\value{
A list with the following components:
  \item{B}{the (p x m x length(ncomp)) matrix containing the regression coefficients. Each row corresponds to a predictor variable and each column to a response variable. The third dimension of the matrix B corresponds to the number of PLS components used to compute the regression coefficients. If ncomp has length 1, B is just a (p x m) matrix.}
  \item{Ypred}{the (ntest x m x length(ncomp)) containing the predicted values of the response variables for the observations from Xtest. The third dimension of the matrix Ypred corresponds to the number of PLS components used to compute the regression coefficients.}
  \item{P}{the (p x max(ncomp)) matrix containing the X-loadings.}
  \item{Q}{the (m x max(ncomp)) matrix containing the Y-loadings.}
  \item{T}{the (ntrain x max(ncomp)) matrix containing the X-scores (latent components)}
  \item{R}{the (p x max(ncomp)) matrix containing the weights used to construct the latent components.}

  }

  \references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{http://www.pnas.org/content/111/14/5117}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2016. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btw705/2667156/KODAMA-an-R-package-for-knowledge-discovery-and}{Link}
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

