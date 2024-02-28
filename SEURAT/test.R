install.packages('Seurat')
# install.packages('radian')
# install.packages('IRkernel')
library(Seurat)
print(getwd())

data <- read.table('./imputations/DS1/DS6_clean.csv', header=TRUE, sep=',')
data_matrix <- as.matrix(data)
rownames(data_matrix) <- data_matrix[, 1]

print("Creating Seurat object")
seurat_object <- CreateSeuratObject(counts = data_matrix, project = 'test', min.cells = 3, max.cells=3, min.features=100)
print("Normalizing data")

#VlnPlot(seurat_object, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)
print("Finding variable features")
seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst", nfeatures = 2000)
# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(seurat_object), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(seurat_object)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

print("Scaling data")
seurat_object <- ScaleData(seurat_object, features = rownames(seurat_object))
print("Running PCA")
seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))
print("Cluster")
seurat_object <- FindNeighbors(seurat_object, dims = 1:10)
seurat_object <- FindClusters(seurat_object, resolution = 0.5)
print("Run TSNE and UMAP")
#seurat_object <- RunTSNE(seurat_object, dims = 1:10)
seurat_object <- RunUMAP(seurat_object, dims = 1:10)
print("Plot TSNE and UMAP")
#DimPlot(seurat_object, reduction = "tsne")
DimPlot(seurat_object, reduction = "umap")