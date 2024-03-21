# required external packages for SIMLR
library("Matrix")
library("parallel")
#library("multiple")
library("aricode")


install.packages("RSpectra", type='source')

# Needed for SIMLR Large Scale
library("Rcpp")
library("pracma")
library("RcppAnnoy")
library("RSpectra")
library("Rtsne")
sourceCpp("./SIMLR/src/Rtsne.cpp")


# load the igraph package to compute the NMI
#install.packages('igraph')
#library("igraph")

print(getwd())
#setwd('./SIMLR')
# load the palettes for the plots
library(grDevices)

# load the SIMLR R package
source("./SIMLR/R/compute.multiple.kernel.R")
source("./SIMLR/R/network.diffusion.R")
source("./SIMLR/R/utils.simlr.R")
source("./SIMLR/R/tsne.R")
source("./SIMLR/R/SIMLR.R")
source("./SIMLR/R/utils.simlr.R")
source("./SIMLR/R/utils.simlr.large.scale.R")
source("./SIMLR/R/SIMLR.Rtsne.R")
source("./SIMLR/R/SIMLR_Large_Scale.R")


# load the C file

# NOTE 1: we make use of an external C program during the computations of SIMLR.
# The code is located in the R directory in the file projsplx_R.c. In order to 
# use SIMLR one needs to compite the program. To do so, one needs to run on the 
# shell the command R CMD SHLIB -c projsplx_R.c. 
# The precompiled projsplx_R.so is already provided for MAC OS X only. 
# If one wants to use SIMLR on other operative systems, the file projsplx_R.so 
# needs to be deleted, and re-compiled. 

# NOTE 2: for Windows, the command dyn.load("./R/projsplx_R.so") needs to be 
# substituted with the command dyn.load("./R/projsplx_R.dll"). 

dyn.load("./SIMLR/R/projsplx_R.so")

# load the datasets
load(file="./SIMLR/data/Test_1_mECS.RData")
load(file="./SIMLR/data/Test_2_Kolod.RData")
load(file="./SIMLR/data/Test_3_Pollen.RData")
load(file="./SIMLR/data/Test_4_Usoskin.RData")

# test SIMLR.R on example 1
set.seed(11111)
cat("Performing analysis for Test_1_mECS","\n")
res_example1 = SIMLR(X=Test_1_mECS$in_X,c=Test_1_mECS$n_clust)
#nmi_1 = compare(Test_1_mECS$true_labs[,1],res_example1$y$cluster,method="nmi")
install.packages("aricode")
library(aricode)
nmi_value <- NMI(Test_1_mECS$true_labs[, 1], res_example1$y$cluster)
print(nmi_value)

# test SIMLR.R on example 2
set.seed(22222)
cat("Performing analysis for Test_2_Kolod","\n")
res_example2 = SIMLR(X=Test_2_Kolod$in_X,c=Test_2_Kolod$n_clust)
nmi_2 = NMI(Test_2_Kolod$true_labs[, 1], res_example2$y$cluster)
print(nmi_2)

set.seed(42)

cat("Trying for DS1")
data <- read.table('./imputations/DS1/DS6_clean.csv', header=TRUE, sep=',')
data_matrix <- as.matrix(data)
ds_example1 = SIMLR_Large_Scale(X=data_matrix, c=9)

shape_file <- './imputations/DS1/expr_shape.csv'
numpy_data <- read.table(shape_file, header=TRUE, sep=',')
true_clusters <- as.matrix(numpy_data)
n <- true_clusters[3]

num_groups <- ncol(data) / n
group_list <- rep(1:num_groups, each = n)

print(num_groups)
names(group_list) <- colnames(numpy_data)

print(group_list)
print(ds_example1$y$cluster)

nmi_ds1 = NMI(group_list, ds_example1$y$cluster)
print(nmi_ds1)

set.seed(1111)
data_noisy <- read.table('./imputations/DS1/DS6_45.csv', header=TRUE, sep=',')
data_matrix_noisy <- as.matrix(data_noisy)
ds_example1_noisy <- SIMLR_Large_Scale(X=data_matrix_noisy, c=9)

shape_file <- './imputations/DS1/expr_shape.csv'
numpy_data <- read.table(shape_file, header=TRUE, sep=',')
true_clusters <- as.matrix(numpy_data)
n <- true_clusters[3]

num_groups <- ncol(data) / n
group_list <- rep(1:num_groups, each = n)

print(num_groups)
names(group_list) <- colnames(df)

print(group_list)
print(ds_example1_noisy$y$cluster)

nmi_ds1_noisy = NMI(group_list, ds_example1_noisy$y$cluster)
print(nmi_ds1_noisy)

# test SIMLR.R on example 3
set.seed(33333)
cat("Performing analysis for Test_3_Pollen","\n")
res_example3 = SIMLR(X=Test_3_Pollen$in_X,c=Test_3_Pollen$n_clust)
nmi_3 = compare(Test_3_Pollen$true_labs[,1],res_example3$y$cluster,method="nmi")

# test SIMLR.R on example 4
set.seed(44444)
cat("Performing analysis for Test_4_Usoskin","\n")
res_example4 = SIMLR(X=Test_4_Usoskin$in_X,c=Test_4_Usoskin$n_clust)
nmi_4 = compare(Test_4_Usoskin$true_labs[,1],res_example4$y$cluster,method="nmi")

# make the scatterd plots
plot(res_example1$ydata,col=c(topo.colors(Test_1_mECS$n_clust))[Test_1_mECS$true_labs[,1]],xlab="SIMLR component 1", ylab="SIMLR component 2",pch=20,main="SIMILR 2D visualization for Test_1_mECS")

plot(res_example2$ydata,col=c(topo.colors(Test_2_Kolod$n_clust))[Test_2_Kolod$true_labs[,1]],xlab="SIMLR component 1", ylab="SIMLR component 2",pch=20,main="SIMILR 2D visualization for Test_2_Kolod")

plot(res_example3$ydata,col=c(topo.colors(Test_3_Pollen$n_clust))[Test_3_Pollen$true_labs[,1]],xlab="SIMLR component 1", ylab="SIMLR component 2",pch=20,main="SIMILR 2D visualization for Test_3_Pollen")

plot(res_example4$ydata,col=c(topo.colors(Test_4_Usoskin$n_clust))[Test_4_Usoskin$true_labs[,1]],xlab="SIMLR component 1", ylab="SIMLR component 2",pch=20,main="SIMILR 2D visualization for Test_4_Usoskin")
