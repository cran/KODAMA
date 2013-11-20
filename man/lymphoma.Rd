\docType{data}

\name{lymphoma}

\alias{lymphoma}

\title{Lymphoma Gene Expression Dataset}

\description{
   This dataset consists of gene expression profiles of the three most prevalent adult lymphoid malignancies: diffuse large B-cell lymphoma (DLBCL), follicular lymphoma (FL), and B-cell chronic lymphocytic leukemia (B-CLL). The dataset consists of 4,682 genes in 62 mRNA samples: 42 samples of DLBCL, 9 samples of FL, and 11 samples. of B-CLL. Missing value are imputed and data are standardized as described in Dudoit, et al. (2002).

}

\usage{data(lymphoma)}

\value{
A list with the following elements:
	\item{data}{Gene expression data. A matrix with 62 rows and 4,682 columns.}
	\item{class}{Class index. A vector with 62 elements.}
  
}


\references{
Alizadeh AA, Eisen MB, Davis RE, \emph{et al.}\cr
Distinct types of diffuse large B-cell lymphoma identified by gene expression profiling. \cr
\emph{Nature} 2000;403(6769):503-511.
\cr
\cr
Dudoit S, Fridlyand J, Speed TP.\cr
Comparison of discrimination methods for the classification of tumors using gene expression data. \cr
\emph{J Am Stat Assoc} 2002;97(417):77-87.
}
\examples{
data(lymphoma)
class=1+as.numeric(as.factor(unlist(lymphoma$class)))
cc=cmdscale(dist(lymphoma$data))
plot(cc,pch=21,bg=class,xlab="First Component",ylab="Second Component");

# WARNING: The next example is high computational extensive
#
# kk=KODAMA(lymphoma$data,FUN=PLS.SVM.CV)
# cc=cmdscale(kk$dissimilarity)
# plot(cc,pch=21,bg=class,xlab="First Component",ylab="Second Component");
}

\keyword{datasets}
