\name{KODAMA.visualization}

\alias{KODAMA.visualization}

\title{Visualization of KODAMA output}

\description{
	Provides a simple function to transform the KODAMA dissimilarity matrix in a low-dimensional space.
}
\usage{
KODAMA.visualization(kk,
                     method=c("t-SNE","MDS","UMAP"),
                     perplexity=min(30,floor((kk$knn_Armadillo$neighbors+1)/3)-1), 
                     ...)
  
}

\arguments{
  \item{kk}{output of \code{\link{KODAMA.matrix}} function.}
  \item{method}{
  	method to be considered for transforming the dissimilarity matrix in a low-dimensional space. Choices are "\code{t-SNE}", "\code{MDS}", and "\code{UMAP}".
}
  \item{perplexity}{
  	Perplexity parameter. (optimal number of neighbors) for "\code{t-SNE}" only.
}
  \item{...}{
  other parameters for "\code{t-SNE}", "\code{MDS}", or "\code{UMAP}".
  }

}


\value{
    The function returns a matrix contains the coordinates of the datapoints in a low-dimensional space.
    
}

\author{Stefano Cacciatore and Leonardo Tenori}

\references{
Cacciatore S, Luchinat C, Tenori L	\cr
Knowledge discovery by accuracy maximization.\cr
\emph{Proc Natl Acad Sci U S A} 2014;111(14):5117-22. doi: 10.1073/pnas.1220873111. \href{https://www.pnas.org/doi/10.1073/pnas.1220873111}{Link}
\cr	\cr
Cacciatore S, Tenori L, Luchinat C, Bennett PR, MacIntyre DA	\cr
KODAMA: an updated R package for knowledge discovery and data mining.	\cr
\emph{Bioinformatics} 2017;33(4):621-623. doi: 10.1093/bioinformatics/btw705. \href{https://academic.oup.com/bioinformatics/article/33/4/621/2667156}{Link}
\cr	\cr
L.J.P. van der Maaten and G.E. Hinton.\cr
Visualizing High-Dimensional Data Using t-SNE. \cr
\emph{Journal of Machine Learning Research} 9 (Nov) : 2579-2605, 2008.
\cr	\cr
L.J.P. van der Maaten. \cr
Learning a Parametric Embedding by Preserving Local Structure. \cr
\emph{In Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics (AISTATS), JMLR W&CP} 5:384-391, 2009.
\cr	\cr
McInnes L, Healy J, Melville J. \cr
Umap: Uniform manifold approximation and projection for dimension reduction. \cr
\emph{arXiv preprint}:1802.03426. 2018 Feb 9.
}


\seealso{\code{\link{KODAMA.visualization}}}

\examples{
\donttest{

 data(iris)
 data=iris[,-5]
 labels=iris[,5]
 kk=KODAMA.matrix(data,FUN="KNN",f.par=2)
 cc=KODAMA.visualization(kk,"t-SNE")
 plot(cc,col=as.numeric(labels),cex=2)

}
}