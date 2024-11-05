## ----echo=FALSE, results='asis'-----------------------------------------------
knitr::kable(matrix(ncol=3,c("kNN","kNN","kNN","PLS-DA","PLS-DA","PLS-DA",2,5,10,2,3,4,9.371,9.362,9.381,9.976,9.933,9.977),dimnames = list(NULL,c("Classifier","parameter","Entropy"))))

## ----echo=FALSE, results='asis'-----------------------------------------------
knitr::kable(matrix(ncol=3,c("kNN","kNN","kNN","kNN","kNN","kNN",
                             "PLS-DA","PLS-DA","PLS-DA","PLS-DA","PLS-DA","PLS-DA",
                             2,3,5,10,15,20,2,5,10,20,50,100,
                             12.847,12.228,12.129,12.432,12.783,13.137,13.370,13.416,13.322,12.637,11.327,11.307),
                    
                    dimnames = list(NULL,c("Classifier","parameter","Entropy"))))

