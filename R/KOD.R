samm = function (d, k = 2) {
  

  x <- as.matrix(d)
  y = cmdscale(d, k)
  z=samm_cpp(x,y,k)
  list(points = z)
}





KODAMA=function (data, M = 100, Tcycle = 20, FUN_VAR = function(x) {
  ceiling(ncol(x))
}, FUN_SAM = function(x) {
  ceiling(nrow(x) * 0.75)
}, bagging = FALSE, FUN = c("KNN","PLS-DA"), f.par = 5, 
W = NULL, constrain = NULL, fix=NULL, 
epsilon = 0.05,dims=2,landmarks=5000) 
{

  if(is.null(fix)) fix = rep(FALSE, nrow(data))
  if(is.null(constrain)) constrain = 1:nrow(data)

  
  data = as.matrix(data)
  shake = FALSE
  nsample=nrow(data) 
  landpoints=NULL
  

 
  LMARK=(nsample>landmarks)
  if(LMARK){
    landpoints=sort(sample(nsample,landmarks))
    Tdata=data[-landpoints,,drop=FALSE]
    Xdata=data[landpoints,,drop=FALSE]
    Tfix=fix[-landpoints]
    Xfix=fix[landpoints] 

    Tconstrain=constrain[-landpoints]
    Xconstrain=constrain[landpoints] 
    vect_proj = matrix(NA,nrow = M, ncol = nrow(Tdata))
  }else{
    Xdata=data
    Xfix=fix
    Xconstrain=constrain  
    landpoints=1:nsample
  }
  
  nva = ncol(Xdata)
  nsa = nrow(Xdata)
  res = matrix(nrow = M, ncol = nsa)
  ma = matrix(0, ncol = nsa, nrow = nsa)
  normalization = matrix(0, ncol = nsa, nrow = nsa)
  FUN_VAR = FUN_VAR(Xdata)
  FUN_SAM = FUN_SAM(Xdata)
  vect_acc = matrix(NA,nrow = M, ncol = Tcycle)
  
  accu = NULL
  whF = which(!Xfix)
  whT = which(Xfix)
  FUN_SAM = FUN_SAM - length(whT)
  pb <- txtProgressBar(min = 0, max = M, style = 1)
  
  for (k in 1:M) {
    setTxtProgressBar(pb, k)

    #
    # Here a number of samples are selected randomly
    #
    sva = sample(nva, FUN_VAR, FALSE, NULL)
    ssa = c(whT, sample(whF, FUN_SAM, bagging, NULL))
#ssa=1:873
    #
    # Here the selection of variables and samples of landmark datapoints is performed
    #   
    x = Xdata[ssa, sva]
    xva = ncol(x)
    xsa = nrow(x)
    Xconstrain_ssa=as.numeric(as.factor(Xconstrain[ssa]))
    Xconstrain_ssa_previous=Xconstrain[ssa]
    Xfix_ssa=Xfix[ssa]

    if(is.null(W)){
      if(xsa<=200){
        XW=Xconstrain_ssa
      }else{
        clust= as.numeric(kmeans(x,50)$cluster)
        tab=apply(table(clust,Xconstrain_ssa),2,which.max)
        XW=as.numeric(as.factor(tab[as.character(Xconstrain_ssa)]))     
      }
    }else{
      XW=W[landpoints][ssa]
      if (any(is.na(XW))) {
        if(xsa<=200){
          unw = unique(XW)
          unw = unw[-which(is.na(unw))]
          ghg = is.na(XW)
          nnew=length(unique(Xconstrain_ssa[ghg]))
          XW[ghg]=as.numeric(as.factor(Xconstrain_ssa[ghg]))+ length(unw)
        }else{
          clust= as.numeric(kmeans(x,50)$cluster)
          tab=apply(table(clust,Xconstrain_ssa),2,which.max)
          constrain_temp=as.numeric(as.factor(tab[as.character(Xconstrain_ssa)]))  
          unw = unique(XW)
          unw = unw[-which(is.na(unw))]
          ghg = is.na(XW)
          nnew=length(unique(constrain_temp[ghg]))
          XW[ghg]=as.numeric(as.factor(constrain_temp[ghg]))+ length(unw)
        }
      }
    }
    clbest=XW
    if(LMARK) {
      xTdata=Tdata[,sva]
    }else{
      xTdata=NULL
    }


    #############################################################################
    #                                                                           #
    #                                 CORE ALGORITHM                            #
    #                                                                           #
    #############################################################################


  if(is.null(xTdata)){
    xTdata=matrix(1,ncol=1,nrow=1)
    proj=1
  }else{
    proj=2
  }
  matchFUN=pmatch(FUN[1],c("KNN","PLS-DA"))
  
  
  

yatta = corecpp(x, xTdata,clbest, Tcycle, matchFUN, f.par, Xconstrain_ssa, Xfix_ssa, shake,proj)


 #   yatta = core_cpp(x,xTdata,clbest,Tcycle,FUN,f.par,Xconstrain_ssa,Xfix_ssa,shake)

    #############################################################################
    #############################################################################
    
    if (!is.list(yatta)) 
      print(yatta)
    if (is.list(yatta)) {
      clbest = as.vector(yatta$clbest)
      accu = yatta$accbest
      yatta$vect_acc=as.vector(yatta$vect_acc)
      yatta$vect_acc[yatta$vect_acc==-1]=NA
      vect_acc[k, ] = yatta$vect_acc
      
      if(LMARK){
        yatta$vect_proj=as.vector(yatta$vect_proj)
        yatta$vect_proj[Tfix]=W[-landpoints][Tfix]
#        tab=as.numeric(apply(table(yatta$vect_proj,Tconstrain),2,which.max))
        
#        tt=table(as.numeric(yatta$clbest),Xconstrain_ssa)
#        tt_constr=rownames(tt)[apply(tt,2,which.max)]
#        names(tt_constr)=colnames(tt)
        
        
#        yatta$vect_proj=as.numeric(tt_constr[as.character(Tconstrain)])
        
        vect_proj[k, ] = yatta$vect_proj
      }
      
      
      uni = unique(clbest)
      nun = length(uni)
      for (ii in 1:nun) ma[ssa[clbest == uni[ii]], 
                           ssa[clbest == uni[ii]]] = 
                        ma[ssa[clbest == uni[ii]], 
                           ssa[clbest == uni[ii]]] + 1
      normalization[ssa, ssa] = normalization[ssa, ssa] +  1
      res[k, ssa] = clbest
    }
  }
  close(pb)
  ma = ma/normalization
  Edist = as.matrix(dist(Xdata))
  ma[ma < epsilon] = 0
  mam = (1/ma) * Edist
  
  mam[is.na(mam)] <- .Machine$double.xmax
  mam[is.infinite(mam) & mam > 0] <- .Machine$double.xmax
  mam = floyd(mam)
  mam[mam == .Machine$double.xmax] <- NA
  
  prox = Edist/mam
  diag(prox) = 1
  prox[is.na(prox)] = 0
  maxvalue=max(mam, na.rm = T)
  mam[is.na(mam)] = maxvalue
  mam = as.dist(mam)
  
  #Calculation of the entropy of the proximity matrix
  y=prox
  diag(y)=NA
  yy=as.numeric(y)
  yy=yy[!is.na(yy)]
  yy=yy/sum(yy)
  H=-sum(ifelse(yy>0,yy*log(yy),0))
  
  if(LMARK){
    pp=matrix(nrow=nsample,ncol=dims)
    mimi=min(mam[mam!=0],na.rm = T)

    pp_landpoints=samm((mam+runif(length(mam))*mimi/1e6),k=dims)$points #


    pr=another(pp_landpoints,Xdata,Tdata,res,vect_proj,epsilon) 
    pp[landpoints,]=pp_landpoints
    pp[-landpoints,]=pr$pp;
  
    total_res=matrix(nrow=M,ncol=nsample)
    total_res[,landpoints]=res
    total_res[,-landpoints]=vect_proj;
  }else{
    mimi=min(mam[mam!=0],na.rm = T)

    pp_landpoints=samm((mam+runif(length(mam))*mimi/1e6),k=dims)$points #
    pp=pp_landpoints
    total_res=res
  }
  
  
  return(list(dissimilarity = mam,
              pp=pp,
              acc = accu, proximity = prox, 
              v = vect_acc, res = total_res, 
              data = Xdata, 
              f.par = f.par,entropy=H,
              landpoints=landpoints))
}
  
  
  




  
  
  
  
  
# This function performs a permutation test to assess association between the 
# KODAMA output and any additional related parameters such as clinical metadata.

k.test = function (data, labels, n = 100) 
{
  data=as.matrix(data)
  if(is.factor(labels)){
    lev=levels(labels)
    Q2Y=try(fit_pls(data,as.matrix(as.numeric(labels)),1),silent = TRUE)
    v=NULL
    for(i in 1:n){
      ss=sample(1:nrow(data))
      v[i]=try(fit_pls(data,as.matrix(as.numeric(labels)[ss]),1),silent = TRUE)
    }
    pval=pnorm(Q2Y, mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
  }else{
    Q2Y=try(fit_pls(data,as.matrix(labels),2),silent = TRUE)
    pval=NULL
    v=NULL
    for(i in 1:n){
      ss=sample(1:nrow(data))
      v[i]=try(fit_pls(data,as.matrix(labels[ss]),2),silent = TRUE)
    }
    
    pval=pnorm(Q2Y, mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
  }
  pval
}



# This function can be used to extract the variable ranking.

loads = function (model,method=c("loadings","kruskal.test")) 
{
  mat=pmatch(method,c("loadings","kruskal.test"))[1]
  nn = nrow(model$res)
  for (i in 1:nn) {
    clu = model$res[i, ]
    na.clu = !is.na(clu)
    clu = clu[na.clu]
    clu=as.numeric(as.factor(clu))
    red.out = matrix(ncol = ncol(model$data), nrow = nn)
    if (length(unique(clu)) > 1) {
      if(mat==1)
         red.out[i, ] = as.numeric(pls.kodama(Xtrain = model$data[na.clu,], 
                                              Xtest  = model$data[na.clu,], 
                                              as.factor(clu), ncomp = 1)$P[, 1])
      if(mat==2)
         red.out[i, ] = apply(model$data,2,function(x) -log(kruskal.test(x[na.clu],as.factor(clu))$p.value))
    }
  }
  colMeans(abs(red.out), na.rm = T)
}




mcplot = function (model){
  A=model$v
  A[,1]=0
  plot(A[1,],type="l",xlim=c(1,ncol(model$v)),ylim=c(0,1),xlab="Numer of interatation",ylab="Accuracy")
  for(i in 1:nrow(A))
      points(A[i,],type="l")
}




core_cpp <- function(x, 
                     xTdata=NULL,
                     clbest, 
                     Tcycle=20, 
                     FUN=c("KNN","PLS-DA"), 
                     fpar=2, 
                     constrain=NULL, 
                     fix=NULL, 
                     shake=FALSE) {
  
  if (is.null(constrain)) 
    constrain = 1:length(clbest)
  
  if (is.null(fix)) 
    fix = rep(FALSE, length(clbest))
  if(is.null(xTdata)){
    xTdata=matrix(1,ncol=1,nrow=1)
    proj=1
  }else{
    proj=2
  }
  matchFUN=pmatch(FUN[1],c("KNN","PLS-DA"))
  
  corecpp(x, xTdata,clbest, Tcycle=20, matchFUN, fpar, constrain, fix, shake,proj)
}







pls.double.cv = function(Xdata,Ydata,constrain=1:nrow(Xdata),compmax=min(c(ncol(Xdata),nrow(Xdata))),perm.test=FALSE){
  Xdata=as.matrix(Xdata)
  constrain=as.numeric(as.factor(constrain))
  if(is.factor(Ydata)){
    lev=levels(Ydata)
    o=double_pls_cv(Xdata,as.matrix(as.numeric(Ydata)),constrain,1,1,compmax)
    o$conf=table(o$Ypred,Ydata)
    o$acc=(sum(diag(o$conf))*100)/length(Ydata)
    Q2Y=o$Q2Y
    pval=NULL
    if(perm.test){
      v=NULL
      pb <- txtProgressBar(min = 0, max = 100, style = 1)
      
      for(i in 1:100){
        setTxtProgressBar(pb, 1)
        ss=sample(1:nrow(Xdata))
        v[i]=double_pls_cv(Xdata[ss,],as.matrix(as.numeric(Ydata)),constrain,1,2,compmax)$Q2Y
      }
      pval=pnorm(Q2Y, mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
    }
    o$Ypred=factor(lev[o$Ypred],levels=lev)
    o$Yfit=factor(lev[o$Yfit],levels=lev)
    o$p.value=pval
    o$R2X=diag((t(o$T)%*%(o$T))%*%(t(o$P)%*%(o$P)))/sum(scale(Xdata,T,T)^2)
  }else{
    o=double_pls_cv(Xdata,as.matrix(Ydata),constrain,2,1,compmax)
    o$Yfit=as.numeric(o$Yfit)
    o$Ypred=as.numeric(o$Ypred)
    Q2Y=o$Q2Y
    pval=NULL
    if(perm.test){
      v=NULL
      for(i in 1:100){
        ss=sample(1:nrow(Xdata))
        v[i]=double_pls_cv(Xdata[ss,],as.matrix(Ydata),constrain,2,2,compmax)$Q2Y
      }
      pval=pnorm(Q2Y, mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
    }
    o$p.value=pval
  }
  o
}





knn.double.cv = function(Xdata,Ydata,constrain=1:nrow(Xdata),compmax=min(c(ncol(Xdata),nrow(Xdata))),perm.test=FALSE){
  Xdata=as.matrix(Xdata)
  constrain=as.numeric(as.factor(constrain))
  if(is.factor(Ydata)){
    lev=levels(Ydata)
    o=double_knn_cv(Xdata,as.numeric(Ydata),constrain,1,1,compmax)
    o$conf=table(o$Ypred,Ydata)
    o$acc=(sum(diag(o$conf))*100)/length(Ydata)
    Q2Y=o$Q2Y
    pval=NULL
    if(perm.test){
      v=NULL
      pb <- txtProgressBar(min = 0, max = 100, style = 1)
      
      for(i in 1:100){
        setTxtProgressBar(pb, 1)
        ss=sample(1:nrow(Xdata))
        v[i]=double_knn_cv(Xdata[ss,],as.numeric(Ydata),constrain,1,2,compmax)$Q2Y
      }
      pval=pnorm(Q2Y, mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
    }
    o$Ypred=factor(lev[o$Ypred],levels=lev)
    o$Yfit=factor(lev[o$Yfit],levels=lev)
    o$p.value=pval
  }else{
    o=double_knn_cv(Xdata,as.numeric(Ydata),constrain,2,1,compmax)
    o$Yfit=as.numeric(o$Yfit)
    o$Ypred=as.numeric(o$Ypred)
    Q2Y=o$Q2Y
    pval=NULL
    if(perm.test){
      v=NULL
      for(i in 1:100){
        ss=sample(1:nrow(Xdata))
        v[i]=double_knn_cv(Xdata[ss,],as.numeric(Ydata),constrain,2,2,compmax)$Q2Y
      }
      pval=pnorm(Q2Y, mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
    }
    o$p.value=pval
  }
  o
}




 