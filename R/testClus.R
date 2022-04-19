##################################################################################################
# Test the Best Partition with CLUS                                                              #
# Copyright (C) 2021                                                                             #
#                                                                                                #
# This code is free software: you can redistribute it and/or modify it under the terms of the    #
# GNU General Public License as published by the Free Software Foundation, either version 3 of   #
# the License, or (at your option) any later version. This code is distributed in the hope       #
# that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    #
# more details.                                                                                  #
#                                                                                                #
# Elaine Cecilia Gatto | Prof. Dr. Ricardo Cerri | Prof. Dr. Mauri Ferrandin                     #
# Federal University of Sao Carlos (UFSCar: https://www2.ufscar.br/) Campus Sao Carlos           #
# Computer Department (DC: https://site.dc.ufscar.br/)                                           #
# Program of Post Graduation in Computer Science (PPG-CC: http://ppgcc.dc.ufscar.br/)            #
# Bioinformatics and Machine Learning Group (BIOMAL: http://www.biomal.ufscar.br/)               #
#                                                                                                #
##################################################################################################


FolderRoot = "~/TCP-KNN-NH/"
FolderScripts = paste(FolderRoot, "/R/", sep="")


##################################################################################################
# FUNCTION MOUNT HYBRID PARTITION                                                                #
#   Objective                                                                                    #
#   Parameters                                                                                   #
#   Return                                                                                       #
#     performance classifier                                                                     #
##################################################################################################
buildAndTest2 <- function(ds, dataset_name, number_dataset,
                         number_folds, number_cores,
                         folderResults, diretorios){

  if(interactive()==TRUE){ flush.console() }

  diretorios = diretorios

  #cat("\nFrom 1 to 10 folds!")
  f = 1
  buildParalel <- foreach(f = 1:number_folds) %dopar%{
  #while(f<=number_folds){

    cat("\n#=========================================================")
    cat("\n#Fold: ", f)
    cat("\n#=========================================================")

    diretorios = diretorios

    FolderRoot = "~/TCP-KNN-NH/"
    FolderScripts = paste(FolderRoot, "/R/", sep="")

    setwd(FolderScripts)
    source("utils.R")

    setwd(FolderScripts)
    source("libraries.R")


    ############################################################################################################
    #cat("\nLOAD FUNCTION CONVERT ARFF \n")
    converteArff <- function(arg1, arg2, arg3){
      str = paste("java -jar ", diretorios$folderUtils, "/R_csv_2_arff.jar ", arg1, " ", arg2, " ", arg3, sep="")
      print(system(str))
      cat("\n")
    }

    ##############################################################################
    constroiParticoes <- function(TotalParticoes){

      data <- data.frame(matrix(NA,    # Create empty data frame
                                nrow = TotalParticoes,
                                ncol = 2))

      names(data) = c("numberPartition", "numberGroup")

      i = 1
      a = 1
      while(i<=nrow(data)){
        data[i,1] = a + 1
        data[i,2] = a + 1
        i = i + 1
        a = a + 1
        gc()
      }

      return(data)

    }

    ########################################################################################
    cat("\nSelect Partition for", f, "\n")
    FolderSplitComm = paste(diretorios$folderCommunities, "/Split-", f, sep="")
    FolderPSplit = paste(diretorios$folderPartitions, "/Split-", f, sep="")
    FolderTempSplit = paste(diretorios$folderTested, "/Split-", f, sep="")
    if(dir.create(FolderTempSplit)==FALSE){dir.create(FolderTempSplit)}

    ########################################################################################
    cat("\nOpen Train file ", f, "\n")
    setwd(diretorios$folderCVTR)
    nome_arq_tr = paste(dataset_name, "-Split-Tr-", f, ".csv", sep="")
    cat("\n\t\t", nome_arq_tr)
    arquivo_tr = data.frame(read.csv(nome_arq_tr))

    ########################################################################################
    cat("\nOpen Validation file ", f, "\n")
    setwd(diretorios$folderCVVL)
    nome_arq_vl = paste(dataset_name, "-Split-Vl-", f, ".csv", sep="")
    cat("\n\t\t", nome_arq_vl)
    arquivo_vl = data.frame(read.csv(nome_arq_vl))

    ########################################################################################
    cat("\nOpen Test file ", f, "\n")
    setwd(diretorios$folderCVTS)
    nome_arq_ts = paste(dataset_name, "-Split-Ts-", f, ".csv", sep="")
    cat("\n\t\t", nome_arq_ts)
    arquivo_ts = data.frame(read.csv(nome_arq_ts))
    arquivo_tr2 = rbind(arquivo_tr, arquivo_vl)

    cat("\nOpen CHOOSED THRESHOLDS")
    setwd(FolderPSplit)
    escolhidos = data.frame(read.csv(paste("fold-",f,"-knn-nh-choosed.csv", sep="")))
    total = nrow(escolhidos)

    u = 1
    while(u<=total){

        cat("\n#=========================================================")
        cat("\n#U = ", u)
        cat("\n#=========================================================")

        diretorios = diretorios

        cat("\nOpen CHOOSED THRESHOLDS")
        setwd(FolderPSplit)
        escolhidos = data.frame(read.csv(paste("fold-",f,"-knn-nh-choosed.csv", sep="")))
        total = nrow(escolhidos)
        escolhidos = escolhidos[u,]

        FolderPartComm = paste(FolderSplitComm, "/", escolhidos$sparsification, sep="")
        FolderPartKnn = paste(FolderPSplit, "/", escolhidos$sparsification, sep="")
        FolderTempKnn = paste(FolderTempSplit, "/", escolhidos$sparsification, sep="")
        if(dir.create(FolderTempKnn)==FALSE){dir.create(FolderTempKnn)}

        setwd(FolderPSplit)
        Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
        total_Knn_NH = nrow(Knn_NH)
        Knn_NH2 = Knn_NH[u,]

        setwd(FolderPartKnn)
        particoes = data.frame(read.csv(paste("knn-",u, "-nh-partition.csv", sep="")))
        nr = nrow(particoes)

        if(nr==0){
          #cat("\não tem partição para validar!")

        } else {

          g = 1
          while(g<=Knn_NH2$numberComm){

            cat("\n#=========================================================")
            cat("\n#Group = ", g)
            cat("\n#=========================================================")

            diretorios = diretorios

            FolderTestGroup = paste(FolderTempKnn, "/Group-", g, sep="")
            if(dir.exists(FolderTestGroup)== FALSE){dir.create(FolderTestGroup) }

            ######################################################################################################################
            cat("\nSpecific Group: ", g, "\n")
            grupoEspecifico = filter(particoes, groups == g)

            ######################################################################################################################
            cat("\nTRAIN: Mount Group ", g, "\n")
            atributos_tr = arquivo_tr2[ds$AttStart:ds$AttEnd]
            n_a = ncol(atributos_tr)
            classes_tr = select(arquivo_tr2, grupoEspecifico$label)
            n_c = ncol(classes_tr)
            grupo_tr = cbind(atributos_tr, classes_tr)
            fim_tr = ncol(grupo_tr)

            ######################################################################################################################
            cat("\n\tTRAIN: Save Group", g, "\n")
            setwd(FolderTestGroup)
            nome_tr = paste(dataset_name, "-split-tr-", f, "-group-", g, ".csv", sep="")
            write.csv(grupo_tr, nome_tr, row.names = FALSE)

            ######################################################################################################################
            cat("\n\tINICIO FIM TARGETS: ", g, "\n")
            inicio = ds$LabelStart
            fim = fim_tr
            ifr = data.frame(inicio, fim)
            write.csv(ifr, "inicioFimRotulos.csv", row.names = FALSE)

            ######################################################################################################################
            cat("\n\tTRAIN: Convert Train CSV to ARFF ", g , "\n")
            nome_arquivo_2 = paste(dataset_name, "-split-tr-", f, "-group-", g, ".arff", sep="")
            arg1Tr = nome_tr
            arg2Tr = nome_arquivo_2
            arg3Tr = paste(inicio, "-", fim, sep="")
            str = paste("java -jar ", diretorios$folderUtils, "/R_csv_2_arff.jar ", arg1Tr, " ", arg2Tr, " ", arg3Tr, sep="")
            print(system(str))

            ######################################################################################################################
            cat("\n\tTRAIN: Verify and correct {0} and {1} ", g , "\n")
            arquivo = paste(FolderTestGroup, "/", arg2Tr, sep="")
            str0 = paste("sed -i 's/{0}/{0,1}/g;s/{1}/{0,1}/g' ", arquivo, sep="")
            print(system(str0))

            ######################################################################################################################
            cat("\n\tTEST: Mount Group: ", g, "\n")
            atributos_ts = arquivo_ts[ds$AttStart:ds$AttEnd]
            classes_ts = select(arquivo_ts, grupoEspecifico$label)
            grupo_ts = cbind(atributos_ts, classes_ts)
            fim_ts = ncol(grupo_ts)
            cat("\n\tTest Group Mounted: ", g, "\n")

            ######################################################################################################################
            cat("\n\tTEST: Save Group ", g, "\n")
            setwd(FolderTestGroup)
            nome_ts = paste(dataset_name, "-split-ts-", f, "-group-", g, ".csv", sep="")
            write.csv(grupo_ts, nome_ts, row.names = FALSE)

            ######################################################################################################################
            cat("\n\tTEST: Convert CSV to ARFF ", g , "\n")
            nome_arquivo_3 = paste(dataset_name, "-split-ts-", f,"-group-", g , ".arff", sep="")
            arg1Ts = nome_ts
            arg2Ts = nome_arquivo_3
            arg3Ts = paste(inicio, "-", fim, sep="")
            str = paste("java -jar ", diretorios$folderUtils, "/R_csv_2_arff.jar ", arg1Ts, " ", arg2Ts, " ", arg3Ts, sep="")
            print(system(str))

            ######################################################################################################################
            cat("\n\tTEST: Verify and correct {0} and {1} ", g , "\n")
            arquivo = paste(FolderTestGroup, "/", arg2Ts, sep="")
            str0 = paste("sed -i 's/{0}/{0,1}/g;s/{1}/{0,1}/g' ", arquivo, sep="")
            cat("\n")
            print(system(str0))
            cat("\n")

            ######################################################################################################################
            cat("\nCreating .s file for clus")
            if(inicio == fim){

              nome_config = paste(dataset_name, "-split-", f, "-group-", g, ".s", sep="")
              sink(nome_config, type = "output")

              cat("[General]")
              cat("\nCompatibility = MLJ08")

              cat("\n\n[Data]")
              cat(paste("\nFile = ", nome_arquivo_2, sep=""))
              cat(paste("\nTestSet = ", nome_arquivo_3, sep=""))

              cat("\n\n[Attributes]")
              cat("\nReduceMemoryNominalAttrs = yes")

              cat("\n\n[Attributes]")
              cat(paste("\nTarget = ", fim, sep=""))
              cat("\nWeights = 1")

              cat("\n")
              cat("\n[Tree]")
              cat("\nHeuristic = VarianceReduction")
              cat("\nFTest = [0.001,0.005,0.01,0.05,0.1,0.125]")

              cat("\n\n[Model]")
              cat("\nMinimalWeight = 5.0")

              cat("\n\n[Output]")
              cat("\nWritePredictions = {Test}")
              cat("\n")
              sink()

              ######################################################################################################################

              cat("\nExecute CLUS: ", g , "\n")
              nome_config2 = paste(FolderTestGroup, "/", nome_config, sep="")
              str = paste("java -jar ", diretorios$folderUtils, "/Clus.jar ", nome_config2, sep="")
              print(system(str))

            } else {

              nome_config = paste(dataset_name, "-split-", f, "-group-", g, ".s", sep="")
              sink(nome_config, type = "output")

              cat("[General]")
              cat("\nCompatibility = MLJ08")

              cat("\n\n[Data]")
              cat(paste("\nFile = ", nome_arquivo_2, sep=""))
              cat(paste("\nTestSet = ", nome_arquivo_3, sep=""))

              cat("\n\n[Attributes]")
              cat("\nReduceMemoryNominalAttrs = yes")

              cat("\n\n[Attributes]")
              cat(paste("\nTarget = ", inicio, "-", fim, sep=""))
              cat("\nWeights = 1")

              cat("\n")
              cat("\n[Tree]")
              cat("\nHeuristic = VarianceReduction")
              cat("\nFTest = [0.001,0.005,0.01,0.05,0.1,0.125]")

              cat("\n\n[Model]")
              cat("\nMinimalWeight = 5.0")

              cat("\n\n[Output]")
              cat("\nWritePredictions = {Test}")
              cat("\n")
              sink()

              cat("\nExecute CLUS: ", g , "\n")
              nome_config2 = paste(FolderTestGroup, "/", nome_config, sep="")
              str = paste("java -jar ", diretorios$folderUtils, "/Clus.jar ", nome_config2, sep="")
              print(system(str))

            }

            ####################################################################################
            cat("\n\nOpen predictions")
            nomeDoArquivo = paste(FolderTestGroup, "/", dataset_name, "-split-", f,"-group-", g, ".test.pred.arff", sep="")
            predicoes = data.frame(foreign::read.arff(nomeDoArquivo))

            ####################################################################################
            cat("\nS\nPLIT PREDICTIS")
            if(inicio == fim){
              #cat("\n\nOnly one label in this group")

              ####################################################################################
              cat("\n\nSave Y_true")
              setwd(FolderTestGroup)
              classes = data.frame(predicoes[,1])
              names(classes) = colnames(predicoes)[1]
              write.csv(classes, "y_true.csv", row.names = FALSE)

              ####################################################################################
              cat("\n\nSave Y_true")
              rot = paste("Pruned.p.", colnames(predicoes)[1], sep="")
              pred = data.frame(predicoes[,rot])
              names(pred) = colnames(predicoes)[1]
              setwd(FolderTestGroup)
              write.csv(pred, "y_predict.csv", row.names = FALSE)

              ####################################################################################
              rotulos = c(colnames(classes))
              n_r = length(rotulos)
              gc()

            } else {

              ####################################################################################
              library("foreign")

              ####################################################################################
              cat("\n\nMore than one label in this group")
              comeco = 1+(fim - inicio)


              ####################################################################################
              cat("\n\nSave Y_true")
              classes = data.frame(predicoes[,1:comeco])
              setwd(FolderTestGroup)
              write.csv(classes, "y_true.csv", row.names = FALSE)


              ####################################################################################
              cat("\n\nSave Y_true")
              rotulos = c(colnames(classes))
              n_r = length(rotulos)
              nomeColuna = c()
              t = 1
              while(t <= n_r){
                nomeColuna[t] = paste("Pruned.p.", rotulos[t], sep="")
                t = t + 1
                gc()
              }
              pred = data.frame(predicoes[nomeColuna])
              names(pred) = rotulos
              setwd(FolderTestGroup)
              write.csv(pred, "y_predict.csv", row.names = FALSE)
              gc()
            } # FIM DO ELSE

            # deleting files
            um = paste(dataset_name, "-split-", f, "-group-", g, ".model", sep="")
            dois = paste(dataset_name, "-split-", f, "-group-", g, ".s", sep="")
            tres = paste(dataset_name, "-split-tr-", f, "-group-", g, ".arff", sep="")
            quatro = paste(dataset_name, "-split-ts-", f, "-group-", g, ".arff", sep="")
            cinco = paste(dataset_name, "-split-tr-", f, "-group-", g, ".csv", sep="")
            seis = paste(dataset_name, "-split-ts-", f, "-group-", g, ".csv", sep="")
            sete = paste(dataset_name, "-split-", f, "-group-", g, ".out", sep="")

            setwd(FolderTestGroup)
            unlink(um, recursive = TRUE)
            unlink(dois, recursive = TRUE)
            unlink(tres, recursive = TRUE)
            unlink(quatro, recursive = TRUE)
            unlink(cinco, recursive = TRUE)
            unlink(seis, recursive = TRUE)
            unlink(sete, recursive = TRUE)

            g = g + 1
            gc()
          } # fim do grupo
        } # fim do else

        u = u + 1
        gc()
      } # end KNN

    #f = f + 1
    gc()
  } # fim do for each


}


##################################################################################################
# FUNCTION SPLITS PREDCTIONS HYBRIDS                                                             #
#   Objective                                                                                    #
#      From the file "test.pred.arff", separates the real labels and the predicted labels to     #
#      generate the confusion matrix to evaluate the partition.                                  #
#   Parameters                                                                                   #
#       ds: specific dataset information                                                         #
#       dataset_name: dataset name. It is used to save files.                                    #
#       number_folds: number of folds created                                                    #
#       DsFolds: folder dataset                                                                  #
#       FolderHybPart: path of hybrid partition validation                                       #
#       FolderHybrid: path of hybrid partition test                                              #
#   Return                                                                                       #
#       true labels and predict labels                                                           #
##################################################################################################
juntaResultados2 <- function(ds, dataset_name, number_dataset,
                             number_folds, number_cores,
                             folderResults, diretorios){

  if(interactive()==TRUE){ flush.console() }

  #cat("\nFrom 1 to 10 folds!")
  # start build partitions
  # do fold 1 até o último fold
  f = 1
  gatherR <- foreach(f = 1:number_folds) %dopar%{
  #while(f<=number_folds){

    cat("\n#=========================================================")
    cat("\n#Fold: ", f)
    cat("\n#=========================================================")

    FolderRoot = "~/TCP-KNN-NH/"
    FolderScripts = paste(FolderRoot, "/R/", sep="")

    setwd(FolderScripts)
    source("utils.R")

    setwd(FolderScripts)
    source("libraries.R")

    ##############################################################################
    constroiParticoes <- function(TotalParticoes){

      data <- data.frame(matrix(NA,    # Create empty data frame
                                nrow = TotalParticoes,
                                ncol = 2))

      names(data) = c("numberPartition", "numberGroup")

      i = 1
      a = 1
      while(i<=nrow(data)){
        data[i,1] = a + 1
        data[i,2] = a + 1
        i = i + 1
        a = a + 1
        gc()
      }

      return(data)

    }

    ########################################################################################
    #cat("\nSelect Partition for", f, "\n")
    FolderPartSplit = paste(diretorios$folderPartitions, "/Split-", f, sep="")
    setwd(FolderPartSplit)
    Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
    total_Knn_NH = nrow(Knn_NH)

    FolderTestedSplit = paste(diretorios$folderTested, "/Split-", f, sep="")
    if(dir.exists(FolderTestedSplit)==FALSE){dir.create(FolderTestedSplit)}

    # KNN
    k = 1
    while(k<=total_Knn_NH){

      cat("\n#=========================================================")
      cat("\n#Threshold = ", k)
      cat("\n#=========================================================")

      FolderPartKnn = paste(FolderPartSplit, "/knn-", k, sep="")
      FolderKnn = paste(FolderTestedSplit, "/knn-", k, sep="")

      setwd(FolderPartSplit)
      Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
      total_Knn_NH = nrow(Knn_NH)
      Knn_NH = Knn_NH[k,]

      ########################################################################################
      apagar = c(0)
      y_true = data.frame(apagar)
      y_pred = data.frame(apagar)

      # GROUP
      g = 1
      while(g<=Knn_NH$numberComm){
          cat("\n#=========================================================")
          cat("\n#Groups: ", g)
          cat("\n#=========================================================")

          FolderGroup = paste(FolderKnn, "/Group-", g, sep="")

          #cat("\n\nGather y_true ", g)
          setwd(FolderGroup)
          #setwd(FolderTG)
          y_true_gr = data.frame(read.csv("y_true.csv"))
          y_true = cbind(y_true, y_true_gr)

          setwd(FolderGroup)
          #setwd(FolderTG)
          #cat("\n\nGather y_predict ", g)
          y_pred_gr = data.frame(read.csv("y_predict.csv"))
          y_pred = cbind(y_pred, y_pred_gr)

          #cat("\n\nDeleting files")
          unlink("y_true.csv", recursive = TRUE)
          unlink("y_predict.csv", recursive = TRUE)
          unlink("inicioFimRotulos.csv", recursive = TRUE)

          g = g + 1
          gc()
        } # FIM DO GRUPO

      #cat("\n\nSave files ", g, "\n")
      setwd(FolderKnn)
      #setwd(FolderKnn)
      y_pred = y_pred[,-1]
      y_true = y_true[,-1]
      write.csv(y_pred, "y_predict.csv", row.names = FALSE)
      write.csv(y_true, "y_true.csv", row.names = FALSE)

      k = k + 1
      gc()
    } # FIM DO KNN

    #f = f + 1
    gc()
  } # fim do foreach


} # fim da função


##################################################################################################
# FUNCTION EVALUATION HYBRID PARTITIONS                                                          #
#   Objective                                                                                    #
#      Evaluates the hybrid partitions                                                           #
#   Parameters                                                                                   #
#       ds: specific dataset information                                                         #
#       dataset_name: dataset name. It is used to save files.                                    #
#       number_folds: number of folds created                                                    #
#       FolderHybrid: path of hybrid partition results                                           #
#   Return                                                                                       #
#       Assessment measures for each hybrid partition                                            #
##################################################################################################
avaliaTest2 <- function(ds, dataset_name, number_dataset,
                       number_folds, number_cores,
                       folderResults, diretorios){

  if(interactive()==TRUE){ flush.console() }

  #cat("\nFrom 1 to 10 folds!")
  # start build partitions
  # do fold 1 até o último fold
  f = 1
  avalParal <- foreach(f = 1:number_folds) %dopar%{
  #while(f<=10){

    cat("\n#=========================================================")
    cat("\n#Fold: ", f)
    cat("\n#=========================================================")

    folders = list()

    FolderRoot = "~/TCP-KNN-NH/"
    FolderScripts = paste(FolderRoot, "/R/", sep="")

    setwd(FolderScripts)
    source("utils.R")

    setwd(FolderScripts)
    source("libraries.R")

    ########################################################################################
    #cat("\nSelect Partition for", f, "\n")
    FolderPartSplit = paste(diretorios$folderPartitions, "/Split-", f, sep="")
    setwd(FolderPartSplit)
    Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
    total_Knn_NH = nrow(Knn_NH)

    FolderTestedSplit = paste(diretorios$folderTested, "/Split-", f, sep="")

    # KNN
    k = 1
    while(k<=total_Knn_NH){

      cat("\n#=========================================================")
      cat("\n#Threshold = ", k)
      cat("\n#=========================================================")

      FolderPartKnn = paste(FolderPartSplit, "/knn-", k, sep="")
      FolderTestedKnn = paste(FolderTestedSplit, "/knn-", k, sep="")

      FolderPartSplit = paste(diretorios$folderPartitions, "/Split-", f, sep="")
      setwd(FolderPartSplit)
      Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
      total_Knn_NH = nrow(Knn_NH)
      Knn_NH = Knn_NH[k,]

      ######################################################################################################################
      #cat("\nSpecific Group: ", k, "\n")
      setwd(FolderPartKnn)
      particoes = data.frame(read.csv(paste("knn-",k,"-nh-partition.csv", sep="")))

      #cat("\nData frame")
      apagar = c(0)
      confMatPartitions = data.frame(apagar)
      partitions = c()

      #cat("\nGet the true and predict lables")
      setwd(FolderTestedKnn)
      #setwd(FolderKnn)
      y_true = data.frame(read.csv("y_true.csv"))
      y_pred = data.frame(read.csv("y_predict.csv"))

      #cat("\nCompute measures multilabel")
      y_true2 = data.frame(sapply(y_true, function(x) as.numeric(as.character(x))))
      y_true3 = mldr_from_dataframe(y_true2 , labelIndices = seq(1,ncol(y_true2 )), name = "y_true2")
      y_pred2 = sapply(y_pred, function(x) as.numeric(as.character(x)))

      #cat("\nSave Confusion Matrix")
      setwd(FolderTestedKnn)
      #setwd(FolderKnn)
      salva3 = paste("Conf-Mat-Fold-", f, "-Knn-", k, ".txt", sep="")
      sink(file=salva3, type="output")
      confmat = multilabel_confusion_matrix(y_true3, y_pred2)
      print(confmat)
      sink()

      #cat("\nCreating a data frame")
      confMatPart = multilabel_evaluate(confmat)
      confMatPart = data.frame(confMatPart)
      names(confMatPart) = paste("Fold-", f, "-Knn-", k, sep="")
      namae = paste("Split-", f, "-knn-", k,"-Evaluated.csv", sep="")
      write.csv(confMatPart, namae)

      #cat("\nDelete files")
      setwd(FolderTestedKnn)
      unlink("y_true.csv", recursive = TRUE)
      unlink("y_predict.csv", recursive = TRUE)

      k = k + 1
      gc()
    } # FIM DO KNN

    #f = f + 1
    gc()

  } # fim do for each

  gc()
  cat("\n##################################################################################################")
  cat("\n# Evaluation Folds: END                                                                          #")
  cat("\n##################################################################################################")
  cat("\n\n\n\n")
}


##############################################################################
buildDataFrame <- function(){

  data <- data.frame(matrix(NA,    # Create empty data frame
                            nrow = 22,
                            ncol = 11))

  measures = c("accuracy","average-precision","clp","coverage","F1","hamming-loss","macro-AUC",
               "macro-F1","macro-precision","macro-recall","margin-loss","micro-AUC","micro-F1",
               "micro-precision","micro-recall","mlp","one-error","precision","ranking-loss",
               "recall","subset-accuracy","wlp")

  data$X1 = measures

  return(data)

}


##################################################################################################
# FUNCTION GATHER EVALUATIONS                                                                    #
#   Objective                                                                                    #
#       Gather metrics for all folds                                                             #
#   Parameters                                                                                   #
#       ds: specific dataset information                                                         #
#       dataset_name: dataset name. It is used to save files.                                    #
#       number_folds: number of folds created                                                    #
#       FolderHybrid: path of hybrid partition results                                           #
#   Return                                                                                       #
#       Assessment measures for all folds                                                        #
##################################################################################################
juntaAvaliacoes2 <- function(ds, dataset_name, number_dataset,
                                number_folds, number_cores,
                                folderResults, diretorios){

  # vector with names
  measures = c("accuracy","average-precision","clp","coverage","F1","hamming-loss","macro-AUC",
               "macro-F1","macro-precision","macro-recall","margin-loss","micro-AUC","micro-F1",
               "micro-precision","micro-recall","mlp","one-error","precision","ranking-loss",
               "recall","subset-accuracy","wlp")

  # from fold = 1 to number_folders
  f = 1
  while(f<=number_folds){

    cat("\n#=========================================================")
    cat("\n#Fold: ", f)
    cat("\n#=========================================================")

    # data frame
    apagar = c(0)
    avaliadoFinal = data.frame(apagar, measures)
    avaliadoKnn = data.frame(apagar, measures)
    folds = c(0)
    threshold = c(0)
    nomesThreshold = c(0)
    nomesFolds = c(0)

    #cat("\nSelect Partition for", f, "\n")
    FolderPartSplit = paste(diretorios$folderPartitions, "/Split-", f, sep="")
    setwd(FolderPartSplit)
    Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
    total_Knn_NH = nrow(Knn_NH)

    FolderTestedSplit = paste(diretorios$folderTested, "/Split-", f, sep="")

    # KNN
    k = 1
    while(k<=total_Knn_NH){

      cat("\n#=========================================================")
      cat("\n#Threshold = ", k)
      cat("\n#=========================================================")

      FolderTestedPart = paste(FolderTestedSplit, "/knn-", k, sep="")

      ######################################################################################################################
      setwd(FolderTestedPart)
      #setwd(FolderKnn)
      str = paste("Split-", f, "-knn-", k, "-Evaluated.csv", sep="")
      avaliado = data.frame(read.csv(str))
      names(avaliado)[1] = "medidas"
      avaliadoKnn = cbind(avaliadoKnn, avaliado[,2])
      nomesThreshold[k] = paste("Fold-", f, "-Knn-", k, sep="")
      names(avaliadoKnn)[k+2] = nomesThreshold[k]
      unlink(str, recursive = TRUE)

      k = k + 1
      gc()
    } # FIM DO KNN

    avaliadoKnn = avaliadoKnn[,-1]
    setwd(FolderTestedSplit)
    write.csv(avaliadoKnn, paste("Evaluated-Fold-", f, ".csv", sep=""), row.names = FALSE)

    f = f + 1
    gc()

  } # end folds

}


organizaAvaliacoes2 <- function(ds, dataset_name, number_dataset,
                               number_folds, number_cores,
                               folderResults, diretorios){

  diretorios = directories(dataset_name, folderResults)

  dfs = list()
  dfs2 = list()

  x = 1
  while(x<=number_folds){
    dfs[[x]] = buildDataFrame()
    x = x + 1
    gc()
  }

   # from fold = 1 to number_folders
  f = 1
  while(f<=number_folds){
    cat("\nFold: ", f)

    ########################################################################################
    FolderPartSplit = paste(diretorios$folderPartitions, "/Split-", f, sep="")
    setwd(FolderPartSplit)
    Knn_NH = data.frame(read.csv(paste("fold-", f, "-knn-nh-choosed.csv", sep="")))
    total_Knn_NH = nrow(Knn_NH)

    FolderTestedSplit = paste(diretorios$folderTested, "/Split-", f, sep="")

    ######################################################################################################################
    #setwd(FolderTempKnn)
    setwd(FolderTestedSplit)
    str = paste("Evaluated-Fold-", f, ".csv", sep="")
    dfs2[[f]] = data.frame(read.csv(str))

    unlink(str, recursive = TRUE)

    f = f + 1
    gc()

  } # end folds

  numCol = ncol(dfs2[[1]])-1

  # vector with names
  measures = c("accuracy","average-precision","clp","coverage","F1","hamming-loss","macro-AUC",
               "macro-F1","macro-precision","macro-recall","margin-loss","micro-AUC","micro-F1",
               "micro-precision","micro-recall","mlp","one-error","precision","ranking-loss",
               "recall","subset-accuracy","wlp")
  apagar = c(0)
  nomesKnn = c()
  nomes = c()

  k = 1
  while(k<=numCol){
    cat("\n\nK: ", k)

    resultado = data.frame(measures, apagar)
    nomesFolds = c()
    nomeKnn1 = paste("Evaluated-10Folds-Knn-", k, ".csv", sep="")
    nomeKnn2 = paste("Mean-10Folds-Knn-", k, ".csv", sep="")
    nomeKnn3 = paste("Median-10Folds-Knn-", k, ".csv", sep="")
    nomeKnn4 = paste("SD-10Folds-Knn-", k, ".csv", sep="")

    f = 1
    while(f<=number_folds){
      cat("\nF: ", f)

      # pegando apenas o fold especifico
      res = data.frame(dfs2[[f]])
      nomesColunas = colnames(res)

      # pegando a partir da segunda coluna
      a = k + 1
      res = res[,a]

      resultado = cbind(resultado, res)
      b = ncol(resultado)
      names(resultado)[b] = nomesColunas[a]

      nomes[f] = paste("Fold-",f,"-Knn-", k, sep="")

      f = f + 1
      gc()
    } # fim do fold


    resultado = data.frame(resultado[,-2])
    setwd(diretorios$folderReport)
    write.csv(resultado, nomeKnn1, row.names = FALSE)

    # calculando a média dos 10 folds para cada medida
    media = data.frame(apply(resultado[,-1], 1, mean))
    media = cbind(measures, media)
    names(media) = c("Measures", "Mean10Folds")
    write.csv(media, nomeKnn2, row.names = FALSE)

    mediana = data.frame(apply(resultado[,-1], 1, median))
    mediana = cbind(measures, mediana)
    names(mediana) = c("Measures", "Median10Folds")
    write.csv(mediana, nomeKnn3, row.names = FALSE)

    dp = data.frame(apply(resultado[,-1], 1, sd))
    dp = cbind(measures, dp)
    names(dp) = c("Measures", "SD10Folds")
    write.csv(dp, nomeKnn4, row.names = FALSE)

    k = k + 1
    gc()
  } # fim do k

}


##################################################################################################
# Please, any errors, contact us: elainececiliagatto@gmail.com                                   #
# Thank you very much!                                                                           #
##################################################################################################
