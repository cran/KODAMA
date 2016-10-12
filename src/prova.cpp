

#include <RcppArmadilloExtensions/sample.h>
#include <math.h>    // math routines
#include "ANN/ANN.h"     // ANN library header
#include <R.h>       // R header
//#include <R_ext/Applic.h>
#include "NN.h"
#include "RcppArmadillo.h"


#include <map>
#include <vector>
#include <iostream>
using namespace std;
using namespace Rcpp;

#if !defined(ARMA_64BIT_WORD)  
  #define ARMA_64BIT_WORD  
#endif 


#if !defined(ARMA_DEFAULT_OSTREAM)
#define ARMA_DEFAULT_OSTREAM Rcpp::Rcout
#endif

void VR_sammon(double *dd, Sint *nn, Sint *kd, double *Y, Sint *niter,
               double *stress, Sint *trace, double *aa, double *tol)
{
  int   i, j, k, m, n = *nn, nd = *kd;
  double *xu, *xv, *e1, *e2;
  double dpj, dq, dr, dt;
  double xd, xx;
  double e, epast, eprev, tot, d, d1, ee, magic = *aa;
  
  xu = Calloc(nd * n, double);
  xv = Calloc(nd, double);
  e1 = Calloc(nd, double);
  e2 = Calloc(nd, double);
  
  epast = eprev = 1.0;
  
  /* Error in distances */
  e = tot = 0.0;
  for (j = 1; j < n; j++)
    for (k = 0; k < j; k++) {
      d = dd[k * n + j];
      if (ISNAN(d)) continue;
      tot += d;
      d1 = 0.0;
      for (m = 0; m < nd; m++) {
        xd = Y[j + m * n] - Y[k + m * n];
        d1 += xd * xd;
      }
      ee = d - sqrt(d1);
      //   if(d1 == 0) error("initial configuration has duplicates");
      e += (ee * ee / d);
    }
    e /= tot;
  // if (*trace) Rprintf("Initial stress        : %7.5f\n", e);
  epast = eprev = e;
  
  /* Iterate */
  for (i = 1; i <= *niter; i++) {
    CORRECT:
    for (j = 0; j < n; j++) {
      for (m = 0; m < nd; m++)
        e1[m] = e2[m] = 0.0;
      for (k = 0; k < n; k++) {
        if (j == k)
          continue;
        dt = dd[k * n + j];
        if (ISNAN(dt)) continue;
        d1 = 0.0;
        for (m = 0; m < nd; m++) {
          xd = Y[j + m * n] - Y[k + m * n];
          d1 += xd * xd;
          xv[m] = xd;
        }
        dpj = sqrt(d1);
        
        /* Calculate derivatives */
        dq = dt - dpj;
        dr = dt * dpj;
        for (m = 0; m < nd; m++) {
          e1[m] += xv[m] * dq / dr;
          e2[m] += (dq - xv[m] * xv[m] * (1.0 + dq / dpj) / dpj) / dr;
        }
      }
      /* Correction */
      for (m = 0; m < nd; m++)
        xu[j + m * n] = Y[j + m * n] + magic * e1[m] / fabs(e2[m]);
    }
    
    /* Error in distances */
    e = 0.0;
    for (j = 1; j < n; j++)
      for (k = 0; k < j; k++) {
        d = dd[k * n + j];
        if (ISNAN(d)) continue;
        d1 = 0.0;
        for (m = 0; m < nd; m++) {
          xd = xu[j + m * n] - xu[k + m * n];
          d1 += xd * xd;
        }
        ee = d - sqrt(d1);
        e += (ee * ee / d);
      }
      e /= tot;
    if (e > eprev) {
      e = eprev;
      magic = magic * 0.2;
      if (magic > 1.0e-3) goto CORRECT;
      //    if (*trace) {
      //      Rprintf("stress after %3d iters: %7.5f\n", i - 1, e);
      //    }
      break;
    }
    magic *= 1.5;
    if (magic > 0.5) magic = 0.5;
    eprev = e;
    
    /* Move the centroid to origin and update */
    for (m = 0; m < nd; m++) {
      xx = 0.0;
      for (j = 0; j < n; j++)
        xx += xu[j + m * n];
      xx /= n;
      for (j = 0; j < n; j++)
        Y[j + m * n] = xu[j + m * n] - xx;
    }
    
    if (i % 10 == 0) {
      //     if (*trace) {
      //        Rprintf("stress after %3d iters: %7.5f, magic = %5.3f\n", i, e, magic);
      //      }
      if (e > epast - *tol)
        break;
      epast = e;
    }
  }
  *stress = e;
  Free(xu);
  Free(xv);
  Free(e1);
  Free(e2);
}


int randomnumber(int mass){
  
  arma::vec xxx = arma::randu(1,1);
  int xxxx=(xxx[0]*(mass-1))+1;
  return xxxx;
}

IntegerVector samplewithoutreplace(IntegerVector yy,int size){
  IntegerVector xx(size);
  int rest=yy.size();
  int it;
  for(int ii=0;ii<size;ii++){
    it=randomnumber(rest)-1;
    xx[ii]=yy[it];
    yy.erase(it);
    rest--;
  }
  return xx;
}


// ===============================
// NEAR NEIGHBOUR CLASSIFIER
// ===============================
//
//
// The function utilizes the Approximate Near Neighbor (ANN) C++ library, 
// which can give the exact near neighbours or (as the name suggests) 
// approximate near neighbours to within a specified error bound.  For more 
// information on the ANN library please visit http://www.cs.umd.edu/~mount/ANN/.
// 
//   Bentley J. L. (1975), Multidimensional binary search trees used 
//   for associative search. Communication ACM, 18:309-517.
//   
//   Arya S. and Mount D. M. (1993), Approximate nearest neighbor searching, 
//   Proc. 4th Ann. ACM-SIAM Symposium on Discrete Algorithms (SODA'93), 271-280.
//   
//   Arya S., Mount D. M., Netanyahu N. S., Silverman R. and Wu A. Y (1998), An 
//   optimal algorithm for approximate nearest neighbor searching, Journal of the
//   ACM, 45, 891-923.

arma::uvec which(LogicalVector x) {
  int a=std::accumulate(x.begin(),x.end(), 0.0);
  arma::uvec w(a);
  int counter=0;
  for(int i = 0; i < x.size(); i++) 
    if(x[i] == 1){
      w[counter]=i;
      counter++;
    }
    return w;
}


arma::mat variance(arma::mat x) {
  int nrow = x.n_rows, ncol = x.n_cols;
  arma::mat out(1,ncol);
  
  for (int j = 0; j < ncol; j++) {
    double mean = 0;
    double M2 = 0;
    int n=0;
    double delta, xx;
    for (int i = 0; i < nrow; i++) {
      n = i+1;
      xx = x(i,j);
      delta = xx - mean;
      mean += delta/n;
      M2 = M2 + delta*(xx-mean);
    }
    out(0,j) = sqrt(M2/(n-1));
  }
  return out;
}


List scalecpp1(arma::mat Xtrain,arma::mat Xtest){
  arma::mat mX=mean(Xtrain,0);
  Xtrain.each_row()-=mX;
  Xtest.each_row()-=mX;
  arma::mat vX=variance(Xtrain);  
  Xtrain.each_row()/=vX;
  Xtest.each_row()/=vX;  
  return List::create(
    Named("Xtrain") = Xtrain,
    Named("Xtest")   = Xtest,
    Named("mean")       = mX,
    Named("sd")       = vX
  ) ;
}


arma::mat scalecpp2(arma::mat Xtrain){
  arma::mat mX=mean(Xtrain,0);
  Xtrain.each_row()-=mX;
  arma::mat vX=variance(Xtrain);  
  Xtrain.each_row()/=vX;
  return Xtrain;
}




double accuracy(arma::ivec cl,arma::ivec cvpred){
  double acc=0;
  for(unsigned i=0;i<cl.size();i++){
    if(cl[i]==cvpred[i])
      acc++;
  }
  acc=acc/cl.size();
  return acc;
  
}


// [[Rcpp::export]]
arma::mat floyd(arma::mat data){
  int n=data.n_cols;
  int i,j,k;
  double temp;
  arma::mat A=data;
  for (i=0; i<n; i++)
    A(i,i) = 0;           
  for (k=0; k<n; k++)
    for (i=0; i<n; i++)
      for (j=0; j<n; j++){
        temp=A(i,k)+A(k,j);
        if (temp < A(i,j))
        {
          A(i,j) = temp;
          
        }
      }
      return A;
}



// [[Rcpp::export]]
arma::imat knn_kodama_c(arma::mat Xtrain,arma::ivec Ytrain,arma::mat Xtest,int k) {
  arma::ivec cla=unique(Ytrain);
  int maxlabel=max(Ytrain);
  double* data = Xtrain.memptr();
  int *label=Ytrain.memptr();
  double* query = Xtest.memptr();
  int D=Xtrain.n_cols;
  int ND=Xtrain.n_rows;
  int NQ=Xtest.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*k;
  int *nn_index= new int[nn];
  double *distances= new double[nn];
  arma::imat Ytest(NQ,k);
  get_NN_2Set(data,query,&D,&ND,&NQ,&k,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);
  // arma::mat result(NQ,k);
  for(int j=0;j<NQ;j++){
      int *lab= new int[k];
    arma::ivec scale(maxlabel);
    scale.zeros();
    for(int i=0;i<k;i++){
      lab[i]=label[nn_index[j*k+i]-1];
      scale(lab[i]-1)=scale(lab[i]-1)+1;
      
      int most_common=-1;
      int value=0;
      for(int h=0;h<maxlabel;h++){
        if(scale(h)>value){
          value=scale(h);
          most_common=h;
        }
      }
      Ytest(j,i)=most_common+1;
    }

  }
  delete [] nn_index;
  delete [] distances;
  return Ytest;
}


// [[Rcpp::export]]
arma::mat knn_kodama_r(arma::mat Xtrain,arma::vec Ytrain,arma::mat Xtest,int k) {
//  arma::ivec cla=unique(Ytrain);
  
//  int maxlabel=max(Ytrain);
  
  double* data = Xtrain.memptr();
  double *label=Ytrain.memptr();
  double* query = Xtest.memptr();
  
  int D=Xtrain.n_cols;
  int ND=Xtrain.n_rows;
  int NQ=Xtest.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*k;
  int *nn_index= new int[nn];
  double *distances= new double[nn];
  arma::mat Ytest(NQ,k);
  get_NN_2Set(data,query,&D,&ND,&NQ,&k,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);

  for(int j=0;j<NQ;j++){
  double *lab= new double[k];
    double media=0;
    for(int i=0;i<k;i++){
      lab[i]=label[nn_index[j*k+i]-1];
      media=media+lab[i];
      Ytest(j,i)=media/(i+1);
    }
  }
  delete [] nn_index;
  delete [] distances;
  return Ytest;
}


// [[Rcpp::export]]
arma::mat samm_cpp(arma::mat x,arma::mat y,int k){
  double* X = x.memptr();
  double* Y = y.memptr();
  int nn=x.n_rows;
  int niter=100;
  double e[1]; e[0]=0;
  int trace=1;
  double magic=0.2;
  double tol=0.0001;
  
  
  
  VR_sammon(X, &nn, &k, Y, &niter,e, &trace, &magic, &tol);
  
  arma::mat result(nn,k);
  for(int j=0;j<nn;j++){
    for(int i=0;i<k;i++){
      result(j,i)=Y[i*nn+j];
    }
  }
  return result;
}



// [[Rcpp::export]]
arma::ivec KNNCV(arma::mat x,arma::ivec cl,arma::ivec constrain,int k) {
  arma::ivec Ytest(x.n_rows);
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
//  IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
        
    IntegerVector v=samplewithoutreplace(frame,xsa_t);

  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  
  
  for (int i=0; i<10; i++) {
  
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::ivec Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);

    
    Ytrain=cl.elem(w9);////////////////
    temp=unique(Ytrain);  //unique(cl(w1));
    if(temp.size()>1){
      Xtrain=x.rows(w9);
      Xtest=x.rows(w1);
      //Ytrain=cl.elem(w9);
      arma::imat temp69=knn_kodama_c(Xtrain,Ytrain,Xtest,k);
      Ytest.elem(w1)=temp69.col(k-1);
      
      
    }else{
      for(unsigned int hh=0;hh<Ytest.size();hh++){
      arma::uvec  hh_w1(1),hh_w9(1);
      unsigned int h1=w1(hh);
      unsigned int h9=w9(0);
      hh_w1(0)=h1;
      hh_w9(0)=h9;
      Ytest.elem(hh_w1)=cl.elem(hh_w9);  ////////ho cambiato w1 con w9
      }
    }
    
  }  
  
  return Ytest;
}






arma::mat pred_pls(arma::mat Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp) {
  
  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  // arma::mat mm=a*b;
  
  //X=Xtrain
  arma::mat X=Xtrain;
  
  
  
  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX=mean(Xtrain,0);
  X.each_row()-=mX;
  Xtest.each_row()-=mX;
  
  //Y=Ytrain
  arma::mat Y=Ytrain;
  
  
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,ncomp);
  VV.zeros();
  
  //  UU<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat UU(n,ncomp);
  UU.zeros();
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,ncomp);
  B.zeros();
  
  // Ypred <- matrix(0,ncol=m,nrow=n)
  arma::cube Ypred(w,m,ncomp);
  Ypred.zeros();  
  
  
  arma::mat qq;
  arma::mat pp;
  arma::mat svd_U;
  arma::vec svd_s;
  arma::mat svd_V;
  arma::mat rr;
  arma::mat tt;
  arma::mat uu;
  arma::mat vv;
  
  
  
  
  // for(a in 1:ncomp){
  for (int a=0; a<ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq    
    
    svd_econ(svd_U,svd_s,svd_V,S,"left");
    rr=svd_U.col( 0 );
    
    
    // tt<-scale(X%*%rr,scale=FALSE)
    tt=X*rr; 
    arma::mat mtt=mean(tt,0);
    tt.each_row()-=mtt;
    
    //tnorm<-sqrt(sum(tt*tt))
    double tnorm=sqrt(sum(sum(tt%tt)));
    
    //tt<-tt/tnorm
    tt/=tnorm;
    
    //rr<-rr/tnorm
    rr/=tnorm;
    
    // pp <- crossprod(X,tt)
    pp=trans(X)*tt;
    
    // qq <- crossprod(Y,tt)
    qq=trans(Y)*tt;
    
    
    //uu <- Y%*%qq
    uu=Y*qq;
    
    //vv<-pp
    vv=pp;
    
    if(a>0){
      //vv<-vv-VV%*%crossprod(VV,pp)
      vv-=VV*(trans(VV)*pp);
      
      
      //uu<-uu-TT%*%crossprod(TT,uu)
      uu-=TT*(trans(TT)*uu);
    }
    
    
    
    
    //vv <- vv/sqrt(sum(vv*vv))
    vv/=sqrt(sum(sum(vv%vv)));
    
    //S <- S-vv%*%crossprod(vv,S)
    S-=vv*(trans(vv)*S);
    
    //RR[,a]=rr
    RR.col(a)=rr;
    TT.col(a)=tt;
    PP.col(a)=pp;
    QQ.col(a)=qq;
    VV.col(a)=vv;
    UU.col(a)=uu;
    
    
    
    B.slice(a)=RR*trans(QQ);
    
    
    Ypred.slice(a)=Xtest*B.slice(a);
    
  } 
  for (int a=0; a<ncomp; a++) {
    arma::mat temp1=Ypred.slice(a);
    temp1.each_row()+=mY;
    Ypred.slice(a)=temp1;
  }  
  
  
  arma::mat sli=Ypred.slice(ncomp-1);
  return sli;
  
  
}





// [[Rcpp::export]]
arma::mat transformy(arma::ivec y){
  int n=y.size();
  int nc=max(y);
  arma::mat yy(n,nc);
  yy.zeros();
  for(int i=0;i<nc;i++){
    for(int j=0;j<n;j++){
      yy(j,i)=((i+1)==y(j));
    }
  }
  
  return yy;
}































// [[Rcpp::export]]
arma::ivec PLSDACV(arma::mat x,arma::ivec cl,arma::ivec constrain,int k) {
  
  arma::mat clmatrix=transformy(cl);
  
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  
  int xsa_t = max(constrain);
  
  IntegerVector frame = seq_len(xsa_t);
//  IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
  

    IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  
  
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    
    w1=find(fold==i);
    w9=find(fold!=i);
    temp=unique(cl(w9));
   
    if(temp.size()>1){
      Xtrain=x.rows(w9);
      Xtest=x.rows(w1);
      
       Ytrain=clmatrix.rows(w9);
      
      
      Ytest.rows(w1)=pred_pls(Xtrain,Ytrain,Xtest,k);
      
      
      
    }else{
      for(unsigned int hh=0; hh<Ytest.n_cols;hh++){
         arma::uvec  hh_w1(1),hh_w9(1);
      unsigned int h1=w1(hh);
      unsigned int h9=w9(0);
      hh_w1(0)=h1;
      hh_w9(0)=h9;
        Ytest.rows(hh_w1)=clmatrix.rows(hh_w9);
      }

    }
  }  
  
  
  
  
  
  
  int mm2=constrain.size();
  arma::ivec pp(mm2);
  
  //min_val is modified to avoid a warning
  double min_val=0;
  min_val++;
  arma::uvec ww;
  for (int i=0; i<mm2; i++) {
    ww=i;
    arma::mat v22=Ytest.rows(ww);
    arma::uword index;                                                                                                                                                                                                                                                                                                                
    min_val = v22.max(index);
    pp(i)=index+1;
  }
  return pp;
}


























// [[Rcpp::export]]
List pls_kodama(arma::mat Xtrain,arma::mat Ytrain,arma::mat Xtest,int ncomp) {
  
  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  // w <-dim(Xtest)[1]
  int w = Xtest.n_rows;
  
  // arma::mat mm=a*b;
  
  //X=Xtrain
  arma::mat X=Xtrain;
  
  
  
  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX=mean(Xtrain,0);
  X.each_row()-=mX;
  Xtest.each_row()-=mX;
  
  //Y=Ytrain
  arma::mat Y=Ytrain;
  
  
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,ncomp);
  VV.zeros();
  
  //  UU<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat UU(n,ncomp);
  UU.zeros();
  
  //  B<-matrix(0,ncol=m,nrow=p)
  arma::cube B(p,m,ncomp);
  B.zeros();
  
  // Ypred <- matrix(0,ncol=m,nrow=n)
  arma::cube Ypred(w,m,ncomp);
  Ypred.zeros();  
  
  
  arma::mat qq;
  arma::mat pp;
  arma::mat svd_U;
  arma::vec svd_s;
  arma::mat svd_V;
  arma::mat rr;
  arma::mat tt;
  arma::mat uu;
  arma::mat vv;
  
  
  
  
  // for(a in 1:ncomp){
  for (int a=0; a<ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq
    svd_econ(svd_U,svd_s,svd_V,S,"left");
    rr=svd_U.col( 0 );
    
    // tt<-scale(X%*%rr,scale=FALSE)
    tt=X*rr; 
    arma::mat mtt=mean(tt,0);
    tt.each_row()-=mtt;
    
    //tnorm<-sqrt(sum(tt*tt))
    double tnorm=sqrt(sum(sum(tt%tt)));
    
    //tt<-tt/tnorm
    tt/=tnorm;
    
    //rr<-rr/tnorm
    rr/=tnorm;
    
    // pp <- crossprod(X,tt)
    pp=trans(X)*tt;
    
    // qq <- crossprod(Y,tt)
    qq=trans(Y)*tt;
    
    
    //uu <- Y%*%qq
    uu=Y*qq;
    
    //vv<-pp
    vv=pp;
    
    if(a>0){
      //vv<-vv-VV%*%crossprod(VV,pp)
      vv-=VV*(trans(VV)*pp);
      
      
      //uu<-uu-TT%*%crossprod(TT,uu)
      uu-=TT*(trans(TT)*uu);
    }
    
    
    
    
    //vv <- vv/sqrt(sum(vv*vv))
    vv/=sqrt(sum(sum(vv%vv)));
    
    //S <- S-vv%*%crossprod(vv,S)
    S-=vv*(trans(vv)*S);
    
    //RR[,a]=rr
    RR.col(a)=rr;
    TT.col(a)=tt;
    PP.col(a)=pp;
    QQ.col(a)=qq;
    VV.col(a)=vv;
    UU.col(a)=uu;
    
    
    
    B.slice(a)=RR*trans(QQ);
    
    
    Ypred.slice(a)=Xtest*B.slice(a);
    
  } 
  for (int a=0; a<ncomp; a++) {
    arma::mat temp1=Ypred.slice(a);
    temp1.each_row()+=mY;
    Ypred.slice(a)=temp1;
  }  
  
  arma::mat Rnorm1(ncomp,ncomp);
  Rnorm1.zeros();
  Rnorm1.diag()=sqrt(sum(RR%RR)); 
  
  arma::mat Rnorm2(ncomp,ncomp);
  Rnorm2.zeros();
  Rnorm2.diag()=1/sqrt(sum(RR%RR)); 
  
  
  arma::mat wM=Rnorm1;
  arma::mat wMi=Rnorm2;
  
  RR=RR*wMi;
  TT=TT*wMi;
  QQ=QQ*wM;
  PP=PP*wM;
  
  return List::create(
    Named("B") = B,
    Named("Ypred")   = Ypred,
    Named("P")       = PP,
    Named("Q")       = QQ,
    Named("T")       = TT,
    Named("R")       = RR,
    Named("meanX")   = mX
  );
}

// [[Rcpp::export]]
int unic(arma::mat x){
  int x_size=x.size();
  for(int i=0;i<x_size;i++)
    if(x(i)!=x(0))
      return 2;
    return 1;
}






// [[Rcpp::export]]
List optim_pls_cv(arma::mat x,arma::mat clmatrix,arma::ivec constrain,int ncomp) {
  
  int nsamples=x.n_rows;
  int nvar=x.n_cols;
  ncomp=min(ncomp,nvar);
  ncomp=min(ncomp,nsamples);
  int ncolY=clmatrix.n_cols;
  arma::cube Ypred(nsamples,ncolY,ncomp); 
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
//  IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
  

    IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    if(unic(clmatrix.rows(w9))==2){
      Xtrain=x.rows(w9);
      
      Xtest=x.rows(w1);
      Ytrain=clmatrix.rows(w9);
      List temp0=scalecpp1(Xtrain,Xtest);
      arma::mat Xtrain1=temp0[0];
      arma::mat Xtest1=temp0[1];
      List pls=pls_kodama(Xtrain1,Ytrain,Xtest1,ncomp);
      arma::cube temp1=pls[1];
      for(int ii=0;ii<w1_size;ii++)  for(int jj=0;jj<ncomp;jj++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk,jj)=temp1(ii,kk,jj);  
    }else{
      for(int ii=0;ii<w1_size;ii++)  for(int jj=0;jj<ncomp;jj++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk,jj)=clmatrix(w1[0],kk);  
    }
  }  
  arma::mat x_scaled=scalecpp2(x);
  List pls2=pls_kodama(x_scaled,clmatrix,x_scaled,ncomp);
  
  arma::cube BB  =pls2[0];
  arma::cube Yfit=pls2[1];
  arma::mat  PP  =pls2[2];
  arma::mat  QQ  =pls2[3];
  arma::mat  TT  =pls2[4];
  arma::mat  RR  =pls2[5];
  
  arma::mat res_pred(nsamples,ncomp),res_fit(nsamples,ncomp);
  arma::vec Q2Y(ncomp);
  arma::vec R2Y(ncomp);
  for(int i=0;i<ncomp;i++){
    arma::mat Ypred_i=Ypred.slice(i);
    arma::mat Yfit_i=Yfit.slice(i);
    
    
    //    arma::mat mYpred=Ypred_i;
    arma::mat mYpred=clmatrix;
    //    arma::mat my_i=mean(Ypred_i,0);
    arma::mat my_i=mean(clmatrix,0);
    mYpred.each_row()-=my_i;
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int k=0;k<ncolY;k++){
      for(int j=0;j<nsamples;j++){
        
        double a1=Ypred_i(j,k);
        double b1=clmatrix(j,k);
        double c1=Yfit_i(j,k);
        double d1=mYpred(j,k);
        
        double arg_TQ=(a1-b1);
        PRESSQ+=arg_TQ*arg_TQ;
        double arg_TR=(c1-b1);
        PRESSR+=arg_TR*arg_TR;
        TSS+=d1*d1;  
      }
    }
    Q2Y(i)=1-PRESSQ/TSS;
    R2Y(i)=1-PRESSR/TSS;
  }

  int optim_c=0;
  for(int i=1;i<ncomp;i++)  if(Q2Y(i)>Q2Y(optim_c)) optim_c=i;
  optim_c++;

  return List::create(
    Named("optim_comp") = optim_c,
    Named("Yfit")       = Yfit,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y,
    Named("B")          = BB,
    Named("P")          = PP,
    Named("Q")          = QQ,
    Named("T")          = TT,
    Named("R")          = RR
  );
  
  
  
}







// [[Rcpp::export]]
List optim_knn_r_cv(arma::mat x,arma::vec clmatrix,arma::ivec constrain,int ncomp) {
  int nsamples=x.n_rows;
  ncomp=min(ncomp,nsamples);
  arma::mat Ypred(nsamples,ncomp); 
  arma::vec Ytest(clmatrix.size());
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
  //IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
  
    IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::vec Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    
    Ytrain=clmatrix.elem(w9);
    int a=1;
    
    int Ytrain_size=Ytrain.size();
    for(int i=0;i<Ytrain_size;i++)
      if(Ytrain(i)!=Ytrain(0))
        a=2;
      if(a==2){ 
        Xtrain=x.rows(w9);
        Xtest=x.rows(w1);
        List temp0=scalecpp1(Xtrain,Xtest);
        arma::mat Xtrain1=temp0[0];
        arma::mat Xtest1=temp0[1];
        arma::mat knn=knn_kodama_r(Xtrain1,Ytrain,Xtest1,ncomp);
        for(int ii=0;ii<w1_size;ii++)  
          for(int jj=0;jj<ncomp;jj++)  
            Ypred(w1[ii],jj)=knn(ii,jj);  
      }else{
        for(int ii=0;ii<w1_size;ii++) for(int jj=0;jj<ncomp;jj++) Ypred(w1[ii],jj)=clmatrix(w1[0]);  
        
      }
  }  
  arma::mat x_scaled=scalecpp2(x);
  arma::mat Yfit=knn_kodama_r(x_scaled,clmatrix,x_scaled,ncomp);
  arma::mat res_pred(nsamples,ncomp),res_fit(nsamples,ncomp);
  arma::vec Q2Y(ncomp);
  arma::vec R2Y(ncomp);
  for(int i=0;i<ncomp;i++){
    arma::vec Ypred_i=Ypred.col(i);
    arma::vec Yfit_i=Yfit.col(i);
    
    arma::vec mYpred=clmatrix;
    double my_i=mean(clmatrix);
    for(int j=0;j<nsamples;j++){
      mYpred(j)=mYpred(j)-my_i;
    }
    
    
    double PRESSQ=0,TSS=0,PRESSR=0;
    
    for(int j=0;j<nsamples;j++){
      double a1=Ypred_i(j);
      double a2=clmatrix(j);
      double b1=Yfit_i(j);
      double d1=mYpred(j);
      double arg_TQ=(a1-a2);
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(b1-a2);
      PRESSR+=arg_TR*arg_TR;
      TSS+=d1*d1;  
      
    }
    Q2Y(i)=1-PRESSQ/TSS;
    R2Y(i)=1-PRESSR/TSS;
  }
  int optim_c=0;
  for(int i=1;i<ncomp;i++)  if(Q2Y(i)>Q2Y(optim_c)) optim_c=i;
  optim_c++;
  return List::create(
    Named("optim_comp") = optim_c,
    Named("Yfit")       = Yfit,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y
    
  ) ;
}


// [[Rcpp::export]]
List optim_knn_c_cv(arma::mat x,arma::ivec clmatrix,arma::ivec constrain,int ncomp) {
  int nsamples=x.n_rows;
  ncomp=min(ncomp,nsamples);
  arma::imat Ypred(nsamples,ncomp); 
  arma::ivec Ytest(clmatrix.size());
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
  //IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
  
  
    IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  
  for (int i=0; i<10; i++) {
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::ivec Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    
    Ytrain=clmatrix.elem(w9);
    int a=1;
    
    int Ytrain_size=Ytrain.size();
    for(int i=0;i<Ytrain_size;i++)
      if(Ytrain(i)!=Ytrain(0))
        a=2;
      if(a==2){ 
        Xtrain=x.rows(w9);
        Xtest=x.rows(w1);
        List temp0=scalecpp1(Xtrain,Xtest);
        arma::mat Xtrain1=temp0[0];
        arma::mat Xtest1=temp0[1];
        arma::imat knn=knn_kodama_c(Xtrain1,Ytrain,Xtest1,ncomp);
        for(int ii=0;ii<w1_size;ii++)  
          for(int jj=0;jj<ncomp;jj++)  
            Ypred(w1[ii],jj)=knn(ii,jj);  
      }else{
        for(int ii=0;ii<w1_size;ii++) for(int jj=0;jj<ncomp;jj++) Ypred(w1[ii],jj)=clmatrix(w1[0]);  
        
      }
  }  
  arma::mat x_scaled=scalecpp2(x);
  arma::imat Yfit=knn_kodama_c(x_scaled,clmatrix,x_scaled,ncomp);
  arma::mat res_pred(nsamples,ncomp),res_fit(nsamples,ncomp);
  arma::vec Q2Y(ncomp);
  arma::vec R2Y(ncomp);
  arma::mat clmatrix2=transformy(clmatrix);
  int ncolY=clmatrix2.n_cols;
  for(int i=0;i<ncomp;i++){
    arma::mat Ypred_i=transformy(Ypred.col(i));
    arma::mat Yfit_i=transformy(Yfit.col(i));
    arma::mat mYpred=clmatrix2;
    
    arma::mat my_i=mean(clmatrix2,0);
    mYpred.each_row()-=my_i;
    
    
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int k=0;k<ncolY;k++){
      for(int j=0;j<nsamples;j++){
        double a1=Ypred_i(j,k);
        double b1=clmatrix2(j,k);
        double c1=Yfit_i(j,k);
        double d1=mYpred(j,k);
        
        double arg_TQ=(a1-b1);
        PRESSQ+=arg_TQ*arg_TQ;
        double arg_TR=(c1-b1);
        PRESSR+=arg_TR*arg_TR;
        TSS+=d1*d1;  
      }
    }
    Q2Y(i)=1-PRESSQ/TSS;
    R2Y(i)=1-PRESSR/TSS;
  }
  int optim_c=0;
  for(int i=1;i<ncomp;i++)  if(Q2Y(i)>Q2Y(optim_c)) optim_c=i;
  optim_c++;
  return List::create(
    Named("optim_comp") = optim_c,
    Named("Yfit")       = Yfit,
    Named("Ypred")      = Ypred,
    Named("Q2Y")        = Q2Y,
    Named("R2Y")        = R2Y
    
  ) ;
}




// [[Rcpp::export]]
List double_pls_cv(arma::mat x,arma::mat y,arma::ivec constrain,int type,int verbose,int compmax) {
  if(verbose==2) Rcpp::Rcout<<".";
  
  arma::mat clmatrix;
  arma::ivec cl;
  arma::vec cl_sup;
  if(type==1){
    cl=as<arma::ivec>(wrap(y));
    clmatrix=transformy(cl);
  }
  if(type==2){
    clmatrix=y;
  }
  int best_comp=compmax;
  int nsamples=x.n_rows;
  int nvar=x.n_cols;
  int ncomp=min(nsamples,nvar);
  ncomp=min(ncomp,compmax);
  
  int ncolY=clmatrix.n_cols;
  arma::mat Ypred(nsamples,ncolY); 
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
  //IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
  
    IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  double mean_nc=0;
  int n_nc=0;
  for (int i=0; i<10; i++) {
    Rcpp::checkUserInterrupt();
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    if(unic(y.rows(w9))==2){
      Xtrain=x.rows(w9);
      arma::ivec constrain_train=constrain(w9);
      Xtest=x.rows(w1);
      Ytrain=clmatrix.rows(w9);
      List temp0=scalecpp1(Xtrain,Xtest);
      arma::mat Xtrain1=temp0[0];
      arma::mat Xtest1=temp0[1];
      
      List optim=optim_pls_cv(Xtrain1,Ytrain,constrain_train,ncomp);
      best_comp=optim[0];
      
      mean_nc+=best_comp;
      if(verbose==1) Rcpp::Rcout<<"Number of component selected (loop #"<<i+1<<"): "<<best_comp<<"\n";
      
      List pls=pls_kodama(Xtrain1,Ytrain,Xtest1,best_comp);
      arma::cube temp1=pls[1];
      
      
      
      for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=temp1(ii,kk,best_comp-1);  
      
      n_nc++;
    }else{
      if(verbose==1) Rcpp::Rcout<<"Number of component selected (loop #"<<i+1<<"): "<<"NA\n";
      
      for(int ii=0;ii<w1_size;ii++)  for(int kk=0;kk<ncolY;kk++)  Ypred(w1[ii],kk)=clmatrix(w1[0],kk);  
    }
    
  }  
  arma::mat x_scaled=scalecpp2(x);
  
  
  int b_comp=round(mean_nc/n_nc);
  if(verbose==1) Rcpp::Rcout<<"Number of component selected for R2y calculation: "<<b_comp<<"\n";
  
  List pls2=pls_kodama(x_scaled,clmatrix,x_scaled,b_comp);
  
  arma::cube BB    = pls2[0];
  arma::cube temp2 = pls2[1];
  arma::mat  PP    = pls2[2];
  arma::mat  QQ    = pls2[3];
  arma::mat  TT    = pls2[4];
  arma::mat  RR    = pls2[5];
  
  arma::mat Yfit(nsamples,ncolY); 
  for(int ii=0;ii<nsamples;ii++)  for(int kk=0;kk<ncolY;kk++)  Yfit(ii,kk)=temp2(ii,kk,b_comp-1);  
  
  arma::vec res_pred(nsamples),res_fit(nsamples);
  double Q2Y;
  double R2Y;
  
  
  
  //  arma::mat mYpred=Ypred;
  
  arma::mat mYpred=clmatrix;
  //  arma::mat my_i=mean(Ypred,0);
  arma::mat my_i=mean(clmatrix,0);
  mYpred.each_row()-=my_i;
  double PRESSQ=0,TSS=0,PRESSR=0;
  for(int k=0;k<ncolY;k++){
    for(int j=0;j<nsamples;j++){
      double a1=Ypred(j,k);
      double b1=clmatrix(j,k);
      double c1=Yfit(j,k);
      double d1=mYpred(j,k);
      
      double arg_TQ=(a1-b1);
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(c1-b1);
      PRESSR+=arg_TR*arg_TR;
      TSS+=d1*d1;
    }
  }
  Q2Y=1-PRESSQ/TSS;
  R2Y=1-PRESSR/TSS;
  
  
  
  if(type==1){
    
    
    //min_val is modified to avoid a warning
    double min_val=0;
    min_val++;
    arma::uvec ww;
    for (int i=0; i<nsamples; i++) {
      ww=i;
      arma::mat v22=Ypred.rows(ww);
      arma::mat v33=Yfit.rows(ww);
      arma::uword index;                                                                                                                                                                                                                                                                                                                
      min_val = v22.max(index);
      res_pred(i)=index+1;
      min_val = v33.max(index);
      res_fit(i)=index+1;
    }
  }
  if(type==2){
    for (int j=0; j<nsamples; j++) {
      res_pred(j)=Ypred(j,0);
      res_fit(j)=Yfit(j,0);
    }
  }
  
  if(b_comp==1){
    
    List pls3=pls_kodama(x_scaled,clmatrix,x_scaled,2);
    
    arma::cube BB2    = pls3[0];
    arma::mat  PP2    = pls3[2];
    arma::mat  QQ2    = pls3[3];
    arma::mat  TT2    = pls3[4];
    arma::mat  RR2    = pls3[5];
    
    BB=BB2;
    PP=PP2;
    QQ=QQ2;
    TT=TT2;
    RR=RR2;
  }
  
  return List::create(
    Named("Yfit")  = res_fit,
    Named("Ypred") = res_pred,
    Named("Q2Y")   = Q2Y,
    Named("R2Y")   = R2Y,
    Named("B")     = BB,
    Named("P")     = PP,
    Named("Q")     = QQ,
    Named("T")     = TT,
    Named("R")     = RR
  ) ;
  
  
  
}


////////////////////////////////////////////////////////////////////////////////////


// [[Rcpp::export]]
List double_knn_cv(arma::mat x,arma::vec yy,arma::ivec constrain,int type,int verbose,int compmax) {
  
  if(verbose==2) Rcpp::Rcout<<".";
  
  int best_comp=compmax;
  int nsamples=x.n_rows;
  int ncomp=min(nsamples,compmax);
  arma::vec Ypred(nsamples); 
  arma::vec Ytest(yy.size());
  int xsa_t = max(constrain);
  IntegerVector frame = seq_len(xsa_t);
  //IntegerVector v=RcppArmadillo::sample<IntegerVector>(frame, xsa_t,FALSE, NumericVector::create() ) ;
  
    IntegerVector v=samplewithoutreplace(frame,xsa_t);
  
  
  int mm=constrain.size();
  arma::ivec fold(mm);
  for (int i=0; i<mm; i++) 
    fold[i]=v[constrain(i)-1]%10;
  double mean_nc=0;
  int n_nc=0;
  
  
  for (int i=0; i<10; i++) {
    
    
    
  
    Rcpp::checkUserInterrupt();
    arma::uvec w1,w9;
    arma::ivec temp;
    arma::mat Xtrain,Xtest;
    arma::mat Ytrain;
    w1=find(fold==i);
    w9=find(fold!=i);
    int w1_size=w1.size();
    Ytrain=yy.elem(w9);
    
    int gg=1;
    int y_size=yy.size();
    for(int j=0;j<y_size;j++)
      if(yy(j)!=yy(0))
        gg=2;
    
      
    if(gg==2){
      Xtrain=x.rows(w9);
      arma::ivec constrain_train=constrain(w9);
      Xtest=x.rows(w1);
        
      List temp0=scalecpp1(Xtrain,Xtest);
      arma::mat Xtrain1=temp0[0];
      arma::mat Xtest1=temp0[1];
      List optim;
      arma::mat knn;
      if(type==1){
        arma::ivec iYtrain=as<arma::ivec>(wrap(Ytrain));
        optim=optim_knn_c_cv(Xtrain1,iYtrain,constrain_train,ncomp);
        best_comp=optim[0];
        mean_nc+=best_comp;
        if(verbose==1) Rcpp::Rcout<<"Number of k selected (loop #"<<i+1<<"): "<<best_comp<<"\n";
        arma::imat iknn=knn_kodama_c(Xtrain1,iYtrain,Xtest1,best_comp);
        knn=as<arma::mat>(wrap(iknn));
      }
      if(type==2){
        optim=optim_knn_r_cv(Xtrain1,Ytrain,constrain_train,ncomp);
        best_comp=optim[0];
        mean_nc+=best_comp;
        if(verbose==1) Rcpp::Rcout<<"Number of k selected (loop #"<<i+1<<"): "<<best_comp<<"\n";
        knn=knn_kodama_r(Xtrain1,Ytrain,Xtest1,best_comp);
      }
      for(int ii=0;ii<w1_size;ii++)  Ypred(w1[ii])=knn(ii,best_comp-1);  
      n_nc++;
    }else{
      if(verbose==1) Rcpp::Rcout<<"Number of component selected (loop #"<<i+1<<"): "<<"NA\n";
      for(int ii=0;ii<w1_size;ii++) Ypred(w1[ii])=yy(w1[0]);  
    }
  }
  
  arma::mat x_scaled=scalecpp2(x);
  int b_comp=round(mean_nc/n_nc);
  if(verbose==1) Rcpp::Rcout<<"Number of k selected for R2y calculation: "<<b_comp<<"\n";
  arma::vec Yfit(nsamples); 
  double Q2Y;
  double R2Y;
  arma::mat knn2;
  
  if(type==1){
    arma::ivec iY=as<arma::ivec>(wrap(yy));
    arma::imat iknn2=knn_kodama_c(x_scaled,iY,x_scaled,b_comp);
    knn2=as<arma::mat>(wrap(iknn2));
    for(int ii=0;ii<nsamples;ii++) Yfit(ii)=knn2(ii,b_comp-1);  
    arma::mat ymatrix=transformy(iY);
    int ncolY=ymatrix.n_cols;
    arma::mat Ypred_i=transformy(as<arma::ivec>(wrap(Ypred)));
    arma::mat Yfit_i=transformy(as<arma::ivec>(wrap(Yfit)));
    arma::mat mYpred=ymatrix;
    arma::mat my_i=mean(ymatrix,0);
    mYpred.each_row()-=my_i;
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int k=0;k<ncolY;k++){
      for(int j=0;j<nsamples;j++){
        double a1=Ypred_i(j,k);
        double b1=ymatrix(j,k);
        double c1=Yfit_i(j,k);
        double d1=mYpred(j,k);
        double arg_TQ=(a1-b1);
        PRESSQ+=arg_TQ*arg_TQ;
        double arg_TR=(c1-b1);
        PRESSR+=arg_TR*arg_TR;
        TSS+=d1*d1;  
      }
    }
    Q2Y=1-PRESSQ/TSS;
    R2Y=1-PRESSR/TSS;
  }
  
  if(type==2){
    knn2=knn_kodama_r(x_scaled,yy,x_scaled,b_comp);
    for(int ii=0;ii<nsamples;ii++) Yfit(ii)=knn2(ii,b_comp-1);  
    arma::vec mYpred=yy;
    double my_i=mean(yy);
    for(int j=0;j<nsamples;j++){
      mYpred(j)=mYpred(j)-my_i;
    }
    double PRESSQ=0,TSS=0,PRESSR=0;
    for(int j=0;j<nsamples;j++){
      double a1=Ypred(j);
      double a2=yy(j);
      double b1=Yfit(j);
      double d1=mYpred(j);
      double arg_TQ=(a1-a2);
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(b1-a2);
      PRESSR+=arg_TR*arg_TR;
      TSS+=d1*d1;  
    }
    Q2Y=1-PRESSQ/TSS;
    R2Y=1-PRESSR/TSS;
  }
  

  
  return List::create(
    Named("Yfit")  = Yfit,
    Named("Ypred") = Ypred,
    Named("Q2Y")   = Q2Y,
    Named("R2Y")   = R2Y
  ) ;
}


/////////////////////////////////////////////




// [[Rcpp::export]]
double fit_pls(arma::mat x,arma::mat y,int type) {
  arma::mat clmatrix;
  arma::ivec cl;
  arma::vec cl_sup;
  if(type==1){
    cl=as<arma::ivec>(wrap(y));
    clmatrix=transformy(cl);
  }
  if(type==2){
    clmatrix=y;
  }
  int nsamples=x.n_rows;
  int nvar=x.n_cols;
  int ncomp=min(nsamples,nvar);
  
  int ncolY=clmatrix.n_cols;
  arma::mat Ypred(nsamples,ncolY); 
  arma::mat Ytest(clmatrix.n_rows,clmatrix.n_cols);



  
  arma::mat x_scaled=scalecpp2(x);
  List pls2=pls_kodama(x_scaled,clmatrix,x_scaled,ncomp);
  
  arma::cube temp2 = pls2[1];

  
  arma::mat Yfit(nsamples,ncolY); 
  for(int ii=0;ii<nsamples;ii++)  for(int kk=0;kk<ncolY;kk++)  Yfit(ii,kk)=temp2(ii,kk,ncomp-1);  
  
  arma::vec res_pred(nsamples),res_fit(nsamples);
  double R2Y;
  
  
//  arma::mat mYpred=Ypred;
arma::mat mYpred=clmatrix;
//  arma::mat my_i=mean(Ypred,0);
arma::mat my_i=mean(clmatrix,0);
  mYpred.each_row()-=my_i;
  double PRESSQ=0,TSS=0,PRESSR=0;
  for(int k=0;k<ncolY;k++){
    for(int j=0;j<nsamples;j++){
      double arg_TQ=(Ypred(j,k)-clmatrix(j,k));
      PRESSQ+=arg_TQ*arg_TQ;
      double arg_TR=(Yfit(j,k)-clmatrix(j,k));
      PRESSR+=arg_TR*arg_TR;
      TSS+=mYpred(j,k)*mYpred(j,k);
    }
  }

  R2Y=1-PRESSR/TSS;

  return R2Y;
}






// [[Rcpp::export]]
List corecpp(arma::mat x,
             arma::mat xTdata,
             arma::ivec clbest,
             int Tcycle,
             int FUN,
             int fpar,
             arma::ivec constrain,
             NumericVector fix,
             bool shake,
             int proj) {

  arma::ivec cvpred=clbest;
  arma::ivec cvpredbest;
  if(FUN==1){  
    cvpredbest=KNNCV(x,clbest,constrain,fpar);
  }
  if(FUN==2){
    cvpredbest=PLSDACV(x,clbest,constrain,fpar);    
  }

  double accbest;

  if (shake == FALSE) {
    accbest = accuracy(clbest,cvpredbest);

  }
  else {
    accbest = 0;
  }
  bool success = FALSE;
  int j = 0;
  //Inizialization of the vector to store the values of accuracy
 
  arma::vec vect_acc(Tcycle);

  vect_acc.fill(-1);

  arma::ivec sup1= unique(constrain);

  int nconc=sup1.size();
  
  NumericVector constrain2=as<NumericVector>(wrap(constrain));
  NumericVector fix2=as<NumericVector>(wrap(fix));

  while (j < Tcycle && !success) {

    Rcpp::checkUserInterrupt();
    j++;
    arma::ivec cl = clbest; 
    IntegerVector sup2=seq_len(nconc);

//    IntegerVector temp=RcppArmadillo::sample<IntegerVector>(sup2,1,FALSE,NumericVector::create());

//    IntegerVector ss=RcppArmadillo::sample<IntegerVector>(sup2,nn_temp,FALSE,NumericVector::create());  

    int nn_temp=randomnumber(nconc);
    IntegerVector ss=samplewithoutreplace(sup2,nn_temp);
    

      
    for (int* k = ss.begin(); k != ss.end(); ++k) {
      LogicalVector sele=((constrain2 == *k) & (fix2!=1));
      double flag = std::accumulate(sele.begin(),sele.end(), 0.0);
      if (flag != 0) {

        arma::uvec whi=which(sele);
        arma::ivec uni=unique(cvpredbest.elem(whi));
        
        
   //     IntegerVector soso=RcppArmadillo::sample<IntegerVector>(as<IntegerVector>(wrap(uni)),1,FALSE,NumericVector::create());
        
        IntegerVector ss_uni=as<IntegerVector>(wrap(uni));
        int nn_uni=ss_uni.size();
        int nn_t=randomnumber(nn_uni)-1;
        IntegerVector soso(1);
        soso[0]=ss_uni[nn_t];
        
  //      int nn_temp=randomnumber(nconc);
  //  IntegerVector ss=samplewithoutreplace(sup2,nn_temp);
    
        
        
        IntegerVector soso2=rep(soso,whi.size());
        cl.elem(whi)=as<arma::ivec>(soso2);

      }
    }
    if(FUN==1){
 
      cvpred=KNNCV(x,cl,constrain,fpar);

    }
    if(FUN==2){

      cvpred=PLSDACV(x,cl,constrain,fpar);      

    }
    double accTOT= accuracy(cl,cvpred);
    if (accTOT > accbest) {
      cvpredbest = cvpred;
      clbest = cl;
      accbest = accTOT;
    }
    vect_acc[j] = accbest;
    if (accTOT == 1) 
      success = TRUE;
  }
  

  
  ///////////////////////////////////////////////////////
  
  
  if(proj==2){

    arma::mat projmat;
    int mm2=xTdata.n_rows;
    arma::ivec pp(mm2); 
    if(FUN==1){
      arma::imat temp70=knn_kodama_c(x,clbest,xTdata,fpar);
      pp=temp70.col(fpar-1);
    }
    if(FUN==2){
      arma::mat lcm=transformy(clbest);
      projmat=pred_pls(x,lcm,xTdata,fpar);
      
      //min_val is modified to avoid a warning
      double min_val=0;
      min_val++;
      arma::uvec ww;
      for (int i=0; i<mm2; i++) {
        ww=i;
        arma::mat v22=projmat.rows(ww);
        arma::uword index;                                                                                                                                                                                                                                                                                                                
        min_val = v22.max(index);
        pp(i)=index+1;
      }
    }
    return List::create(Named("clbest") = clbest,
                        Named("accbest") = accbest,
                        Named("vect_acc") = vect_acc,
                        Named("vect_proj") = pp
    );
    
  }else{
    return List::create(Named("clbest") = clbest,
                        Named("accbest") = accbest,
                        Named("vect_acc") = vect_acc
    );
  }
  
}






// [[Rcpp::export]]
List another(arma::mat pptrain,arma::mat xtrain,
             arma::mat xtest,arma::mat res,
             arma::mat Xlink,double epsilon) {
  
  int kkk=5;
  double* knndata = xtrain.memptr();
  double* query = xtest.memptr();
  int dims=pptrain.n_cols;
  int D=xtrain.n_cols;
  int ND=xtrain.n_rows;
  int NQ=xtest.n_rows;
  int Dh=Xlink.n_rows;
  double EPS=0;
  int SEARCHTYPE=1;
  int USEBDTREE=0;
  double SQRAD=0;
  int nn=NQ*kkk;

  int *nn_index= new int[nn];
  double *distances= new double[nn];
  get_NN_2Set(knndata,query,&D,&ND,&NQ,&kkk,&EPS,&SEARCHTYPE,&USEBDTREE,&SQRAD,nn_index,distances);
  
  arma::mat a(NQ,kkk);
  arma::mat b(NQ,kkk);
  arma::mat pptest(NQ,dims);
  for(int j=0;j<NQ;j++){
    int jkkk=j*kkk;
    for(int i=0;i<kkk;i++){
      a(j,i)=nn_index[j*kkk+i];
      int indi=nn_index[j*kkk+i];
      double temp1=0;
      double temp2=0;
      for(int h=0;h<Dh;h++){
        if(Xlink(h,j)==res(h,indi-1)) temp1++;
                if(res(h,indi-1)==NA) temp2++;
      }
      temp2=Dh-temp2;
      double ratio=(temp1/temp2);
      a(j,i)=distances[jkkk+i]*ratio;
      if(ratio<epsilon) temp1=0; 
      if(temp1==0) a(j,i)=-1;
      b(j,i)=nn_index[jkkk+i];
    }
    for(int g=0;g<dims;g++){
      pptest(j,g)=pptrain(nn_index[jkkk]-1,g);
    }
  }
  arma::mat best_pptest=pptest;
  for(int j=0;j<NQ;j++){
    double best=DBL_MAX;
    double temp=0;
    for(int loop=0;loop<1000;loop++){
      double distance=0;
        for(int i=0;i<kkk;i++){
          double dista=0;
          for(int g=0;g<dims;g++){
            double tt1=(pptest(j,g)-pptrain(b(j,i)-1,g));
            dista+=(tt1*tt1);
          }
          dista=sqrt(dista);
          double tt2;
          if(a(j,i)!=-1){
            tt2=(dista-a(j,i));
            distance+=abs(tt2);
          }
        
        }
        if(distance<best){
          best_pptest.row(j)=pptest.row(j);
          best=distance;
          temp=distance;
        }
        arma::mat add;
        add=arma::randn(dims)*temp/100;
        pptest.row(j)=best_pptest.row(j)+add.t();
      }
    
    }
 
  delete [] nn_index;
  delete [] distances;

  return List::create(Named("distances") = a,
                      Named("index") = b,
                      Named("pp") = best_pptest
  );
  
  
}



