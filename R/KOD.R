txtsummary = function (x, digits = 0, scientific = FALSE, range=c("IQR","95%CI")) 
{
  matchFUN=pmatch(range[1],c("IQR","95%CI"))
  if(is.na(matchFUN))
    stop("The range to be considered must be \"IQR\" or \"95%CI\".")
  
  m = median(x, na.rm = TRUE)
  
  if(matchFUN==1)
    ci = quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
  if(matchFUN==2)
    ci = quantile(x, probs = c(0.025, 0.975), na.rm = TRUE)
  if (scientific) {
    m = format(m, digits = digits, scientific = scientific)
    ci = format(ci, digits = digits, scientific = scientific)
  }
  else {
    m = round(m, digits = digits)
    ci = round(ci, digits = digits)
  }
  txt = paste(m, " [", ci[1], " ", ci[2], "]", sep = "")
  txt
}


multi_analysis = function (data, 
                           y, 
                           FUN=c("continuous.test","correlation.test"), ...) 
{
  
  matchFUN = pmatch(FUN[1], c("continuous.test", "correlation.test"))
  if (is.na(matchFUN)) 
    stop("The function to be considered must be  \"continuous.test\" or \"correlation.test\".")
  
  if(matchFUN==1){
    FUN=continuous.test
  }
  if(matchFUN==2){
    FUN=correlation.test
  }
  
  da = NULL
  pval = NULL
  for (i in 1:ncol(data)) {
    sel.na=!is.na(data[, i])
    if(sum(sel.na)>5){
      temp = FUN(name = colnames(data)[i], x = data[sel.na, i], y = y[sel.na], ...)
      da = rbind(da, temp$text)
      pval[i] = temp$p.value
    }else{
      da = rbind(da, c(colnames(data)[i],NA,NA))
      pval[i] = NA
    }
  }
  FDR = p.adjust(pval, method = "fdr")
  FDR = format(FDR, digits = 3, scientific = TRUE)
  da = cbind(da, FDR)
  da
}
#-----for numerical data 

continuous.test = function (name,
                            x,    
                            y,
                            digits = 3,
                            scientific = FALSE, 
                            range = c("IQR","95%CI"), 
                            logchange = TRUE,pos=1,method=c("non-parametric","parametric"), ...) 
{
  
  
  matchFUN = pmatch(method[1], c("non-parametric","parametric"))
    
  if (matchFUN != 1 & matchFUN != 2) {
    stop("Method argument should one of \"non-parametric\",\"parametric\"")
  }
      
      
  
  y = as.factor(y)
  ll = levels(y)
  A = x[y == ll[1]]
  B = x[y == ll[2]]
  nn = length(levels(y))
  v = data.frame()
  v[1, 1] = name
  if (nn == 2) {
    if(matchFUN==1){    
      pval = wilcox.test(x ~ y)$p.value
    }    
    if(matchFUN==2){    
      pval = t.test(x~y)$p.value
    }
    fc = -log2(mean(A, na.rm = TRUE)/mean(B, na.rm = TRUE))
  }
  if (nn > 2) {
    if(matchFUN==1){
      pval = kruskal.test(x ~ y)$p.value
    }
    if(matchFUN==2){
      pval = summary.aov(aov(x~y))[[1]]$`Pr(>F)`[1]
    }
    logchange=FALSE
  }
  if (nn > 1) {
    v[1, 2:(1 + nn)] = tapply(x, y, function(x) txtsummary(x, 
                                                           digits = digits, scientific = scientific, range = range))
    v[1, nn + 2] = txtsummary(x, digits = digits, scientific = scientific)
    v[1, nn + 3] = format(pval, digits = 3, scientific = TRUE)
  }
  else {
    v[1, nn + 3] = NA
  }
  
  
  matchFUN = pmatch(range[1], c("IQR", "95%CI"))
  if(pos==1){
    
    if (matchFUN == 1) {
      names(v) = c("Feature", paste(levels(y), ", median [IQR]", 
                                    sep = ""), "Total, median [IQR]", "p-value")
    }
    if (matchFUN == 2) {
      names(v) = c("Feature", paste(levels(y), ", median [95%CI]", 
                                    sep = ""), "Total, median [95%CI]", "p-value")
    }
  }else{
    
    if (matchFUN == 1) {
      v[1, 1] =  paste(name, ", median [IQR]",sep = "")
    }
    if (matchFUN == 2) {
      v[1, 1] = paste(name, ", median [95%CI]",sep = "")
    }
    names(v) = c("Feature", levels(y), "Total", "p-value")
  }
  v[v == "NA [NA NA]"] = "-"
  if (logchange == TRUE) {
    v = cbind(v[1, 1:(nn + 2)], logchange = round(fc, digits = 2),v[1, (nn + 3)])
    names(v)[nn + 4]="p-value"
    list(text = v, p.value = pval, logchange = fc)
  }
  else {
    list(text = v, p.value = pval)
  }
}

#-----for categorical data 
categorical.test = 
  function (name, x, y) 
  {
    y = as.factor(y)
    nn = length(levels(y))
    t0 = table(x, y)
    ta = cbind(t0, as.matrix(table(x)))
    tb = sprintf("%.1f", t(t(ta)/colSums(ta)) * 100)
    tc = matrix(paste(ta, " (", tb, ")", sep = ""), 
                ncol = nn + 1)
    tc[, c(colSums(t0), -1) == 0] = "-"
    v = NULL
    if (nrow(t0) == 1) {
      p.value = NA
      v[nn + 3] = ""
    }
    else {
      p.value = fisher.test(t0, workspace = 10^7)$p.value
      v[nn + 3] = format(p.value, digits = 3, scientific = TRUE)
    }
    v[1] = name
    group = paste("   ", rownames(ta), ", n (%)", 
                  sep = "")
    cc = cbind(group, tc, rep(NA, length(group)))
    cc = rbind(v, cc)
    colnames(cc) = c("Feature", colnames(t0), "Total", 
                     "p-value")
    cc[is.na(cc)] = ""
    list(text = cc, p.value = p.value)
  }





correlation.test= function(x,y,method = c("pearson", "spearman","MINE"), name=NA, perm=100 , ...){
  matchFUN = pmatch(method[1], c("pearson", "spearman","MINE"))
  if (is.na(matchFUN)) 
    stop("The method to be considered must be  \"pearson\", \"spearman\" or \"MINE\".")
  res=list()
  sel=!is.na(x) & !is.na(y)
  x=x[sel]
  y=y[sel]
  
  res$text = data.frame()
  res$text[1, 1] = name
  res$text[1,2]=NA
  res$text[1,3]=NA
  if(length(x)<5){
    warning("The number of correlated elements is less than 5.")
    res$estimate=NA
    res$p.value=NA
    
  }else{
    if(matchFUN==1){
      temp=cor.test(x,y,method="pearson")
      res$estimate=temp$estimate
      res$p.value=temp$p.value
    }
    if(matchFUN==2){
      temp=cor.test(x,y,method="spearman")
      res$estimate=temp$estimate
      res$p.value=temp$p.value
    }
    if(matchFUN==3){
      res$estimate=mine(x,y)$MIC
      v=NULL
      for(i in 1:perm){
        v[i]=mine(x,sample(y))$MIC
      }
      res$p.value=pnorm(res$estimate, mean = mean(v), sd = sqrt(((length(v) - 
                                                                    1)/length(v)) * var(v)), lower.tail = FALSE)
    }
    res$text[1,2]=round(res$estimate,digits=2)
    res$text[1,3]=format(res$p.value, digits = 3, scientific = TRUE)
  }
  if(matchFUN==1){
    names(res$text)=c("Feature","r","p-value")
  }  
  if(matchFUN==2){
    names(res$text)=c("Feature","rho","p-value")
  }  
  if(matchFUN==3){
    names(res$text)=c("Feature","MIC","p-value")
  }
  
  return(res)
}



pca = function(x,...){
  res=prcomp(x,...)
  ss=sprintf("%.1f",summary(res)$importance[2,]*100)
  res$txt = paste(names(summary(res)$importance[2,])," (",ss,"%)",sep="")
  res
}
















samm = function (d, k = 2) {
  

  x <- as.matrix(d)
  y = cmdscale(d, k)
  z=samm_cpp(x,y,k)
  list(points = z)
}


knnsampling = function(data,n,k=10,cycle=1){
  
  ss=sample(nrow(data),n)
  for(ii in 1:cycle){
    s=ss
    
    data_sampled=data[s,]
    g=knn_Armadillo(data,data_sampled,10)
    gi=g$nn_index
    best=Inf
    ss=s
    for(j in 1:n){
      gij=gi[j,]
      gg=knn_Armadillo(data_sampled, data[gij,],k = k)
      tot=rowSums(gg$distances)
      sel=gij[which.max(tot)]
      ss[j]=sel
    }
  }
  ss
}



KODAMA=function (data, M = 100, Tcycle = 20, FUN_VAR = function(x) {
  ceiling(ncol(x))
}, FUN_SAM = function(x) {
  ceiling(nrow(x) * 0.75)
}, bagging = FALSE, FUN = c("PLS-DA","KNN"), f.par = 5, 
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
    landpoints=sort(knnsampling(data,landmarks))
    #landpoints=sort(sample(nrow(data),landmarks))
    Tdata=data[-landpoints,,drop=FALSE]
    Xdata=data[landpoints,,drop=FALSE]
    Xdata_landpoints=Xdata
    Tfix=fix[-landpoints]
    Xfix=fix[landpoints] 

    Tconstrain=constrain[-landpoints]
    Xconstrain=constrain[landpoints] 
    vect_proj = matrix(NA,nrow = M, ncol = nrow(Tdata))
  }else{
    Xdata=data
    Xdata_landpoints=Xdata
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
  pb <- txtProgressBar(min = 1, max = M, style = 1)
  
  for (k in 1:M) {
    setTxtProgressBar(pb, k)
    
    
    
    
    
    
    if(LMARK){
     ks=round(nsample/landmarks)
      tt=knn_Armadillo(data,data[landpoints,],k=ks)$nn_index
      landpoints2=landpoints
      for(ii in 1:landmarks){
        landpoints2[ii]=tt[ii,sample(ks,1)]
      }
    #landpoints=sort(sample(nrow(data),landmarks))
      Xdata=data[landpoints2,,drop=FALSE]
    }
    
    
    
    
    #
    # Here a number of samples are selected randomly
    #
    sva = sample(nva, FUN_VAR, FALSE, NULL)
    ssa = c(whT, sample(whF, FUN_SAM, bagging, NULL))

    #
    # Here the selection of variables and samples of landmark datapoints is performed
    #   
    x = Xdata[ssa, sva]
    xva = ncol(x)
    xsa = nrow(x)
    Xconstrain_ssa=as.numeric(as.factor(Xconstrain[ssa]))
    Xconstrain_ssa_previous=Xconstrain[ssa]
    Xfix_ssa=Xfix[ssa]
        

    ###################################
    del_n=rep(NA,nrow(x))
    for(ik in 1:(nrow(x)-1)){
      if(is.na(del_n[ik])){
        
        del_n[ik]=ik
        
        for(ij in 2:nrow(x)){
          if(all(x[ik,]==x[ij,])) del_n[ij]=ik
        }
        
        
        
      }
    }
    if(is.na(del_n[nrow(x)])) del_n[nrow(x)]=nrow(x)
    
    
    xsa_same_point=length(unique(del_n))
    #####################################
    
    
    
    
    if(is.null(W)){
      if(xsa_same_point<=200 || length(unique(x))<50){
        XW=Xconstrain_ssa
      }else{
        
        clust= as.numeric(kmeans(x,50)$cluster)
        tab=apply(table(clust,Xconstrain_ssa),2,which.max)
        XW=as.numeric(as.factor(tab[as.character(Xconstrain_ssa)]))
        
      }
    }else{
      XW=W[landpoints][ssa]
    
      if (any(is.na(XW))) {
        if(xsa_same_point<=200 || length(unique(x))<50){

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
    yatta = core_cpp(x,xTdata,clbest,Tcycle,FUN,f.par,Xconstrain_ssa,Xfix_ssa,shake)

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
  Edist = as.matrix(dist(Xdata_landpoints))
  ma[ma < epsilon] = 0
  mam = (1/ma) * Edist
  
  mam[is.na(mam)] <- .Machine$double.xmax
  mam[is.infinite(mam) & mam > 0] <- .Machine$double.xmax
  mam = floyd(mam)
  mam[mam == .Machine$double.xmax] <- NA
  
  prox = Edist/mam
  diag(prox) = 1
  prox[is.na(prox)] = 0
  maxvalue=max(mam, na.rm = TRUE)
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
    mimi=min(mam[mam!=0],na.rm = TRUE)

    pp_landpoints=samm((mam+runif(length(mam))*mimi/1e6),k=dims)$points #


    pr=another(pp_landpoints,Xdata_landpoints,Tdata,res,vect_proj,epsilon) 
    pp[landpoints,]=pp_landpoints
    pp[-landpoints,]=pr$pp;
  
    total_res=matrix(nrow=M,ncol=nsample)
    total_res[,landpoints]=res
    total_res[,-landpoints]=vect_proj;
  }else{
    mimi=min(mam[mam!=0],na.rm = TRUE)

    pp_landpoints=samm((mam+runif(length(mam))*mimi/1e6),k=dims)$points #
    pp=pp_landpoints
    total_res=res
  }
  
  
  return(list(dissimilarity = mam,
              pp=pp,
              acc = accu, proximity = prox, 
              v = vect_acc, res = total_res, 
              data = Xdata_landpoints, 
              f.par = f.par,entropy=H,
              landpoints=landpoints))
}
  
  
  





  
# This function performs a permutation test to assess association between the 
# KODAMA output and any additional related parameters such as clinical metadata.

#k.test = function (data, labels, n = 100) 
#{
#  data=as.matrix(data)
#  compmax=min(dim(data))
#  option=2-as.numeric(is.factor(label))
#  w_R2Y=NULL
#  for(i in 1:n){
#    w_R2Y[i]=double_pls_cv(data,as.matrix(as.numeric(labels)),1:nrow(data),option,2,compmax,1,1)$R2Y
#  }
#  v_R2Y=NULL
#  for(i in 1:n){
#    ss=sample(1:nrow(data))
#    v_R2Y[i]=double_pls_cv(data,as.matrix(as.numeric(labels[ss])),1:nrow(data),option,2,compmax,1,1)$R2Y
#  }
#  pval=wilcox.test(w_R2Y,v_R2Y,alternative = "greater")$p.value
#  pval
#}
k.test = 
  function (data, labels, n = 100) 
  {
    data = as.matrix(data)
    compmax = min(dim(data))
    option = 2 - as.numeric(is.factor(labels))
    
    w_R2Y = pls.double.cv(data, labels, 1:nrow(data),compmax = 2,perm.test = FALSE,times = 1,runn=1)$medianR2Y
    
    v_R2Y = NULL
    for (i in 1:n) {
      ss = sample(1:nrow(data))
      v_R2Y[i] = pls.double.cv(data, labels[ss], 1:nrow(data),compmax = 2,perm.test = FALSE,times = 1,runn=1)$medianR2Y
    }
    pval = sum(v_R2Y>w_R2Y)/n
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
  colMeans(abs(red.out), na.rm = TRUE)
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
                     FUN=c("PLS-DA","KNN"), 
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
  if(is.na(matchFUN))
    stop("The classifier to be considered must be  \"PLS-DA\" or \"KNN\".")
  
  out=corecpp(x, xTdata,clbest, Tcycle, matchFUN, fpar, constrain, fix, shake,proj)
  return(out)
}







pls.double.cv = function(Xdata,
                         Ydata,
                         constrain=1:nrow(Xdata),
                         compmax=min(c(ncol(Xdata),nrow(Xdata))),
                         perm.test=FALSE,
                         optim=TRUE,
                         scaling=c("centering","autoscaling"),
                         times=100,
                         runn=10){

  
  scal=pmatch(scaling,c("centering","autoscaling"))[1]
  optim=as.numeric(optim)
  Xdata=as.matrix(Xdata)
  constrain=as.numeric(as.factor(constrain))
  res=list()
  Q2Y=NULL
  R2Y=NULL
  bcomp=NULL
  if(is.factor(Ydata)){
    lev=levels(Ydata)
    

    for(j in 1:runn){

      o=double_pls_cv(Xdata,as.matrix(as.numeric(Ydata)),constrain,1,2,compmax,optim,scal)
      bcomp[j]=o$bcomp
      o$Ypred=factor(lev[o$Ypred],levels=lev)
      o$conf=table(o$Ypred,Ydata)
      o$acc=(sum(diag(o$conf))*100)/length(Ydata)
      o$Yfit=factor(lev[o$Yfit],levels=lev)
      o$R2X=diag((t(o$T)%*%(o$T))%*%(t(o$P)%*%(o$P)))/sum(scale(Xdata,TRUE,TRUE)^2)
      Q2Y[j]=o$Q2Y
      R2Y[j]=o$R2Y
      res$results[[j]]=o
      
    }
    res$Q2Y=Q2Y
    res$R2Y=R2Y
    res$medianR2Y=median(R2Y)
    res$CI95R2Y=as.numeric(quantile(R2Y,c(0.025,0.975)))
    res$medianQ2Y=median(Q2Y)
    res$CI95Q2Y=as.numeric(quantile(Q2Y,c(0.025,0.975)))
    res$bcomp=floor(median(bcomp,na.rm = TRUE))
    
    bb=NULL;for(h in 1:runn) bb[h]=res$results[[h]]$bcomp
    run = which(bb==res$bcomp)[1]
    
    res$T=res$results[[run]]$T
    res$Q=res$results[[run]]$Q
    res$P=res$results[[run]]$P
    res$B=res$results[[run]]$B
    
    mpred=matrix(ncol=runn,nrow=nrow(Xdata));
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Ypred)
    res$Ypred=apply(mpred,1,function(x) names(which.max(table(x))))
    
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Yfit)
    res$Yfit=apply(mpred,1,function(x) names(which.max(table(x))))
    
    if(perm.test){

      v=NULL
   
      for(i in 1:times){
        ss=sample(1:nrow(Xdata))
        w=NULL
        for(ii in 1:runn)
          w[ii]=double_pls_cv(Xdata[ss,],as.matrix(as.numeric(Ydata)),constrain,1,2,compmax,optim,scal)$Q2Y
        
        v[i]=median(w)
      }
      pval=pnorm(median(Q2Y), mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
      res$Q2Ysampled=v
  #    res$p.value=wilcox.test(Q2Y,v,alternative = "greater")$p.value     
      res$p.value=pval
      
    }
  
  }else{

    for(j in 1:runn){

      o=double_pls_cv(Xdata,as.matrix(Ydata),constrain,2,2,compmax,optim,scal)
      bcomp[j]=o$bcomp
      o$Yfit=as.numeric(o$Yfit)
      o$Ypred=as.numeric(o$Ypred)
      o$R2X=diag((t(o$T)%*%(o$T))%*%(t(o$P)%*%(o$P)))/sum(scale(Xdata,TRUE,TRUE)^2)
      Q2Y[j]=o$Q2Y
      R2Y[j]=o$R2Y
      res$results[[j]]=o
    }
    res$Q2Y=Q2Y
    res$Q2Y=Q2Y
    res$R2Y=R2Y
    res$medianR2Y=median(R2Y)
    res$CI95R2Y=as.numeric(quantile(R2Y,c(0.025,0.975)))
    res$medianQ2Y=median(Q2Y)
    res$CI95Q2Y=as.numeric(quantile(Q2Y,c(0.025,0.975)))
    res$bcomp=floor(median(bcomp,na.rm = TRUE))
    
    
    bb=NULL;for(h in 1:runn) bb[h]=res$results[[h]]$bcomp
    run = which(bb==res$bcomp)[1]
    
    res$T=res$results[[run]]$T
    res$Q=res$results[[run]]$Q
    res$P=res$results[[run]]$P
    res$B=res$results[[run]]$B
    
    mpred=matrix(ncol=runn,nrow=nrow(Xdata));
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Ypred)
    res$Ypred=apply(mpred,1,function(x) median(x))
    
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Yfit)
    res$Yfit=apply(mpred,1,function(x) median(x))
    
    pval=NULL
    if(perm.test){

      v=NULL
      
      for(i in 1:times){
        ss=sample(1:nrow(Xdata))
        w=NULL
        for(ii in 1:runn)

          w[ii]=double_pls_cv(Xdata[ss,],as.matrix(Ydata),constrain,2,2,compmax,optim,scal)$Q2Y

        v[i]=median(w)
      }
      pval=pnorm(median(Q2Y), mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
      res$Q2Ysampled=v
      #    res$p.value=wilcox.test(Q2Y,v,alternative = "greater")$p.value     
      res$p.value=pval
    }

  }
  res$txtQ2Y=txtsummary(res$Q2Y,digits=2)
  res$txtR2Y=txtsummary(res$R2Y,digits=2)
  res
}





knn.double.cv = function(Xdata,
                         Ydata,
                         constrain=1:nrow(Xdata),
                         compmax=min(c(ncol(Xdata),nrow(Xdata))),
                         perm.test=FALSE,
                         optim=TRUE,
                         scaling=c("centering","autoscaling"),
                         times=100,
                         runn=10){

  scal=pmatch(scaling,c("centering","autoscaling"))[1]
  optim=as.numeric(optim)
  Xdata=as.matrix(Xdata)
  constrain=as.numeric(as.factor(constrain))
  
  res=list()
  Q2Y=NULL
  R2Y=NULL
  bk=NULL
  
  if(is.factor(Ydata)){
    lev=levels(Ydata)

    for(j in 1:runn){

      o=double_knn_cv(Xdata,as.numeric(Ydata),constrain,1,2,compmax,optim,scal)
      o$conf=table(o$Ypred,Ydata)
      o$acc=(sum(diag(o$conf))*100)/length(Ydata)
      o$Yfit=factor(lev[o$Yfit],levels=lev)
      o$Ypred=factor(lev[o$Ypred],levels=lev)
      Q2Y[j]=o$Q2Y
      R2Y[j]=o$R2Y
      bk[j]=o$bk
      res$results[[j]]=o
      
      

    }
    
    res$Q2Y=Q2Y
    res$R2Y=R2Y
    res$medianR2Y=median(R2Y)
    res$CI95R2Y=as.numeric(quantile(R2Y,c(0.025,0.975)))
    res$medianQ2Y=median(Q2Y)
    res$CI95Q2Y=as.numeric(quantile(Q2Y,c(0.025,0.975)))
    res$bk=floor(median(bk,na.rm = TRUE))
    
    mpred=matrix(ncol=runn,nrow=nrow(Xdata));
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Ypred)
    res$Ypred=apply(mpred,1,function(x) names(which.max(table(x))))
    
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Yfit)
    res$Yfit=apply(mpred,1,function(x) names(which.max(table(x))))
    
    
    pval=NULL
    if(perm.test){
      
      v=NULL
      
      for(i in 1:times){
        ss=sample(1:nrow(Xdata))
        w=NULL
        for(ii in 1:runn)
          
          w[ii]=double_knn_cv(Xdata[ss,],as.numeric(Ydata),constrain,1,2,compmax,optim,scal)$Q2Y
        
        v[i]=median(w)
      }

      
      pval=pnorm(median(Q2Y), mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
      res$Q2Ysampled=v
      #    res$p.value=wilcox.test(Q2Y,v,alternative = "greater")$p.value     
      res$p.value=pval
    }
    

    
    
  
  }else{

    for(j in 1:runn){
 
      o=double_knn_cv(Xdata,as.numeric(Ydata),constrain,2,2,compmax,optim,scal)
      o$Yfit=as.numeric(o$Yfit)
      o$Ypred=as.numeric(o$Ypred)
      Q2Y[j]=o$Q2Y
      R2Y[j]=o$R2Y
      bk[j]=o$bk
      res$results[[j]]=o
    }
    res$Q2Y=Q2Y
    res$R2Y=R2Y
    res$medianR2Y=median(R2Y)
    res$CI95R2Y=as.numeric(quantile(R2Y,c(0.025,0.975)))
    res$medianQ2Y=median(Q2Y)
    res$CI95Q2Y=as.numeric(quantile(Q2Y,c(0.025,0.975)))
    res$bk=floor(median(bk,na.rm = TRUE))
    
    mpred=matrix(ncol=runn,nrow=nrow(Xdata));
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Ypred)
    res$Ypred=apply(mpred,1,function(x) median(x))
    
    for(h in 1:runn) mpred[,h]= as.vector(res$results[[h]]$Yfit)
    res$Yfit=apply(mpred,1,function(x) median(x))
    
    pval=NULL
    if(perm.test){

      
      v=NULL
      
      for(i in 1:times){
        ss=sample(1:nrow(Xdata))
        w=NULL
        for(ii in 1:runn)
          
          w[ii]=double_knn_cv(Xdata[ss,],as.numeric(Ydata),constrain,2,2,compmax,optim,scal)$Q2Y
        
        v[i]=median(w)
      }

      pval=pnorm(median(Q2Y), mean=mean(v), sd=sqrt(((length(v)-1)/length(v))*var(v)), lower.tail=FALSE) 
      res$Q2Ysampled=v
      #    res$p.value=wilcox.test(Q2Y,v,alternative = "greater")$p.value     
      res$p.value=pval
    }
    

  }
  res$txtQ2Y=txtsummary(res$Q2Y,digits=2)
  res$txtR2Y=txtsummary(res$R2Y,digits=2)
  res
}






















frequency_matching = function(data,label,times=5,seed=1234){

  data=as.data.frame(data)
  data2=data
  for(i in 1:ncol(data2)){
    if(is.numeric(data2[,i])){
    
    v <- quantile(data2[,i],prob=seq(0,0.99,0.2))
    data2[,i]= findInterval(data2[,i], v)
    }
  }
  if(is.null(rownames(data2))){
    rownames(data2)=paste("S",1:nrow(data2),sep="")
    rownames(data)=paste("S",1:nrow(data),sep="")
  }
  names(label)=rownames(data2)
  data2=as.matrix(data2[!is.na(label),])
  label=label[!is.na(label)]
  
  minor=names(which.min(table(label)))
  major=names(which.max(table(label)))
  data_minor=data2[label==minor,,drop=FALSE]
  data_major=data2[label==major,,drop=FALSE]
  
  nc=ncol(data2)
  grid=list()
  count=list()
  rest=list()
  for(j in 1:nc){
    
    lis=list()
    
    h=1
    for(i in j:nc){
      lis[[h]]=levels(as.factor(data2[,i]))
      h=h+1
    }
    grid[[j]]=as.matrix(expand.grid(lis))
    
    co = apply(grid[[j]],1,function(y) sum(apply(as.matrix(data_minor)[,j:nc,drop=FALSE],1,function(x) all(y==x))))
    count[[j]]=co*times
  }
  rest=list()
  rest[[1]]=count[[1]]
  selected=rep(FALSE,nrow(data_major))
  names(selected)=rownames(data_major)
  for(j in 1:nc){
    if(sum(rest[[j]])>0){
      for(i in 1:nrow(grid[[j]])){
        if(rest[[j]][i]!=0){
          who=apply(as.matrix(data_major[,j:nc]),1,function(x) all(grid[[j]][i,]==x))
          n_who=min(sum(who[!selected]),rest[[j]][i])
          rest[[j]][i]=rest[[j]][i]-n_who
          set.seed(seed)
          ss=sample(names(which(who[!selected])),n_who)
          selected[ss]=TRUE
        }
      }
      if(j<nc){
        temp=list()
        for(ii in 2:ncol(grid[[j]]))   temp[[ii-1]]=as.matrix(grid[[j]])[,ii]
        rest[[j+1]]=aggregate(rest[[j]], by=temp, FUN=sum, na.rm=TRUE)[,"x"]
      }else{
        rest[[j+1]]=sum(rest[[j]])
      }
      
    }
  }
  if(sum(rest[[j]])>0){
    set.seed(seed)
    ss=sample(which(!selected),rest[[j+1]])
    selected[ss]=TRUE
  }
  
  selection=c(rownames(data_major[selected,,drop=FALSE]),rownames(data_minor))
  
  
  data=data[selection,]
  data2=data2[selection,]
  label=label[selection]
  return(list(data=data,label=label,selection=selection))#,grid=grid,rest=rest))
}






vis.KODAMA =
  function (data, res, landmarks = 1000, epsilon = 0.05, knnsampling = FALSE,dims=2) 
  {
    
    if(any(is.na(res))){
      landpoints=which(is.na(colSums(res)))
      landmarks=length(landpoints)
    }else{
      
      landmarks=min(landmarks,nrow(data))
      LMARK= FALSE
      if(landmarks<nrow(data)) LMARK =TRUE
      
      if(knnsampling){
        landpoints = sort(knnsampling(data, landmarks))
      }else{
        landpoints = sort(sample(nrow(data), landmarks))
      }
    }
    
    
    nsample=ncol(res)
    M=nrow(res)
    Tdata = data[-landpoints, , drop = FALSE]
    Xdata = data[landpoints, , drop = FALSE]
    Xdata_landpoints = Xdata
    vect_proj = res[,-landpoints]
    vect_landpoints = res[,landpoints]
    
    
    nva = ncol(Xdata)
    nsa = nrow(Xdata)
    
    
    ma = matrix(0, ncol = nsa, nrow = nsa)
    normalization = matrix(0, ncol = nsa, nrow = nsa)
    ####################################
    for(i in 1:M){
      clbest = vect_landpoints[i,]
      
      uni = unique(clbest)
      nun = length(uni)
      for (ii in 1:nun) {
        sel=which(clbest == uni[ii])
        ma[sel,sel] = ma[sel,sel] + 1
      }
      sel2=!is.na(clbest)
      normalization[sel2,sel2] = normalization[sel2,sel2] + 1
    }
    ###########################
    ma = ma/normalization
    Edist = as.matrix(dist(Xdata_landpoints))
    ma[ma < epsilon] = 0
    
    mam = (1/ma) * Edist
    mam[is.na(mam)] <- .Machine$double.xmax
    mam[is.infinite(mam) & mam > 0] <- .Machine$double.xmax
    mam = floyd(mam)
    mam[mam == .Machine$double.xmax] <- NA
    prox = Edist/mam
    diag(prox) = 1
    prox[is.na(prox)] = 0
    maxvalue = max(mam, na.rm = TRUE)
    mam[is.na(mam)] = maxvalue
    mam = as.dist(mam)
    
    if (LMARK) {
      pp = matrix(nrow = nsample, ncol = dims)
      mimi = min(mam[mam != 0], na.rm = TRUE)
      pp_landpoints = samm((mam + runif(length(mam)) * mimi/1e+06), 
                           k = dims)$points
      pr = another(pp_landpoints, Xdata_landpoints, Tdata, 
                   vect_landpoints, vect_proj, epsilon)
      pp[landpoints, ] = pp_landpoints
      pp[-landpoints, ] = pr$pp
      
    }  else {
      mimi = min(mam[mam != 0], na.rm = T)
      pp_landpoints = samm((mam + runif(length(mam)) * mimi/1e+06), 
                           k = dims)$points
      pp = pp_landpoints
      
    }
    pp
  }

 