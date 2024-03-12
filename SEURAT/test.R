install.packages('Seurat')
install.packages('patchwork')
install.packages('ggplot2')
install.packages('fpc')
# install.packages('radian')
# install.packages('IRkernel')
library(Seurat)
library(patchwork)
library(ggplot2)
library(cluster)
library(dplyr)
library(fpc)
print(getwd())

data <- read.table('./imputations/DS1/DS6_clean.csv', header=TRUE, sep=',')
data_matrix <- as.matrix(data)
rownames(data_matrix) <- data_matrix[, 1]

print("Creating Seurat object")
seurat_object <- CreateSeuratObject(counts = data_matrix, project = 'test', min.cells = 3, min.features=100)
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
seurat_object <- FindClusters(seurat_object, resolution = 0.75)
print("Run TSNE and UMAP")
tsne_obj <- RunTSNE(seurat_object, dims = 1:10, check_duplicates = FALSE)
umap_obj <- RunUMAP(seurat_object, dims = 1:10)
print("Plot TSNE and UMAP")
tsne_plot <- DimPlot(tsne_obj, reduction = "tsne") 
#tsne_plot + plot_annotation(title = 'DS1 tSNE')
umap_plot <- DimPlot(umap_obj, reduction = "umap") 
#umap_plot + plot_annotation(title = 'DS1 UMAP')

cluster_ids <- tsne_obj@meta.data$seurat_clusters
tsne1 <- tsne_obj@reductions$tsne@cell.embeddings[,1]
tsne2 <- tsne_obj@reductions$tsne@cell.embeddings[,2]
tsne_data <- data.frame(cluster_id = cluster_ids, x = tsne1, y = tsne2)
centroids <- aggregate(cbind(x, y) ~ cluster_id, data = tsne_data, FUN = mean)

numeric_cluster_ids <- as.numeric(as.character(tsne_data$cluster_id))
silhouette_scores <- silhouette(numeric_cluster_ids, dist(tsne_data[,c("x", "y")]))
sil_df <- data.frame(cluster = numeric_cluster_ids, 
                     silhouette_width = silhouette_scores[, "sil_width"])

avg_sil_width_by_cluster <- sil_df %>%
  group_by(cluster) %>%
  summarise(avg_silhouette_width = mean(silhouette_width))

print(avg_sil_width_by_cluster)

centroids <- aggregate(cbind(x, y) ~ cluster_id, data = tsne_data, FUN = mean)

wcss_per_cluster <- sapply(unique(tsne_data$cluster_id), function(cluster) {
  points_in_cluster <- tsne_data[tsne_data$cluster_id == cluster, c("x", "y")]
  centroid <- centroids[centroids$cluster_id == cluster, c("x", "y")]
  
  points_matrix <- as.matrix(points_in_cluster)
  centroid_matrix <- as.matrix(centroid[rep(1, nrow(points_in_cluster)), ])
  
  sum(rowSums((points_matrix - centroid_matrix)^2))
})

names(wcss_per_cluster) <- unique(tsne_data$cluster_id)
wcss_per_cluster <- wcss_per_cluster[order(as.numeric(names(wcss_per_cluster)))]
res_df <- data.frame(
  cluster_id = as.numeric(names(wcss_per_cluster)),
  avg_sil_width = avg_sil_width_by_cluster$avg_silhouette_width,
  WCSS = wcss_per_cluster
)
print(res_df)

sigmas <- sapply(unique(tsne_data$cluster_id), function(cluster) {
  points <- tsne_data[tsne_data$cluster_id == cluster, c("x", "y")]
  centroid <- centroids[centroids$cluster_id == cluster, c("x", "y")]
  
  points_mat <- as.matrix(points)
  centroid_mat <- as.matrix(centroid[rep(1, nrow(points)), ])
  
  d_squared <- rowSums((points_mat - centroid_mat)^2)
  
  mean(sqrt(d_squared))
})

c_dists <- as.matrix(dist(centroids[, c("x", "y")]))

db_index <- mean(sapply(1:length(unique(tsne_data$cluster_id)), function(i) {
  max((sigmas[i] + sigmas[-i]) / c_dists[i, -i])
}))

# Print the Davies-Bouldin Index
print(db_index)

tsne_plot + geom_point(data = centroids, aes(x = x, y = y), shape = 4, color = "black", size = 5) + ggtitle('DS1 tSNE')




## ============== DS2 ================
data <- read.table('./imputations/DS2/DS6_clean.csv', header=TRUE, sep=',')
data_matrix <- as.matrix(data)
rownames(data_matrix) <- data_matrix[, 1]

print("Creating Seurat object")
seurat_object <- CreateSeuratObject(counts = data_matrix, project = 'test', min.cells = 3, min.features=100)
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
seurat_object <- FindClusters(seurat_object, resolution = 0.135)
print("Run TSNE and UMAP")
tsne_obj <- RunTSNE(seurat_object, dims = 1:10, check_duplicates = FALSE)
umap_obj <- RunUMAP(seurat_object, dims = 1:10)
print("Plot TSNE and UMAP")
tsne_plot <- DimPlot(tsne_obj, reduction = "tsne") 
#tsne_plot + plot_annotation(title = 'DS1 tSNE')
umap_plot <- DimPlot(umap_obj, reduction = "umap") 
#umap_plot + plot_annotation(title = 'DS1 UMAP')

cluster_ids <- tsne_obj@meta.data$seurat_clusters
tsne1 <- tsne_obj@reductions$tsne@cell.embeddings[,1]
tsne2 <- tsne_obj@reductions$tsne@cell.embeddings[,2]
tsne_data <- data.frame(cluster_id = cluster_ids, x = tsne1, y = tsne2)
centroids <- aggregate(cbind(x, y) ~ cluster_id, data = tsne_data, FUN = mean)

numeric_cluster_ids <- as.numeric(as.character(tsne_data$cluster_id))
silhouette_scores <- silhouette(numeric_cluster_ids, dist(tsne_data[,c("x", "y")]))
sil_df <- data.frame(cluster = numeric_cluster_ids, 
                     silhouette_width = silhouette_scores[, "sil_width"])

avg_sil_width_by_cluster <- sil_df %>%
  group_by(cluster) %>%
  summarise(avg_silhouette_width = mean(silhouette_width))

print(avg_sil_width_by_cluster)

centroids <- aggregate(cbind(x, y) ~ cluster_id, data = tsne_data, FUN = mean)

wcss_per_cluster <- sapply(unique(tsne_data$cluster_id), function(cluster) {
  points_in_cluster <- tsne_data[tsne_data$cluster_id == cluster, c("x", "y")]
  centroid <- centroids[centroids$cluster_id == cluster, c("x", "y")]
  
  points_matrix <- as.matrix(points_in_cluster)
  centroid_matrix <- as.matrix(centroid[rep(1, nrow(points_in_cluster)), ])
  
  sum(rowSums((points_matrix - centroid_matrix)^2))
})

names(wcss_per_cluster) <- unique(tsne_data$cluster_id)
wcss_per_cluster <- wcss_per_cluster[order(as.numeric(names(wcss_per_cluster)))]
res_df <- data.frame(
  cluster_id = as.numeric(names(wcss_per_cluster)),
  avg_sil_width = avg_sil_width_by_cluster$avg_silhouette_width,
  WCSS = wcss_per_cluster
)
print(res_df)

sigmas <- sapply(unique(tsne_data$cluster_id), function(cluster) {
  points <- tsne_data[tsne_data$cluster_id == cluster, c("x", "y")]
  centroid <- centroids[centroids$cluster_id == cluster, c("x", "y")]
  
  points_mat <- as.matrix(points)
  centroid_mat <- as.matrix(centroid[rep(1, nrow(points)), ])
  
  d_squared <- rowSums((points_mat - centroid_mat)^2)
  
  mean(sqrt(d_squared))
})

c_dists <- as.matrix(dist(centroids[, c("x", "y")]))

db_index <- mean(sapply(1:length(unique(tsne_data$cluster_id)), function(i) {
  max((sigmas[i] + sigmas[-i]) / c_dists[i, -i])
}))

# Print the Davies-Bouldin Index
print(db_index)

tsne_plot + geom_point(data = centroids, aes(x = x, y = y), shape = 4, color = "black", size = 5) + ggtitle('DS2 tSNE')





## =============== DS3 =================
data <- read.table('./imputations/DS3/DS6_clean.csv', header=TRUE, sep=',')
data_matrix <- as.matrix(data)
rownames(data_matrix) <- data_matrix[, 1]

print("Creating Seurat object")
seurat_object <- CreateSeuratObject(counts = data_matrix, project = 'test', min.cells = 3, min.features=100)
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
seurat_object <- FindClusters(seurat_object, resolution = 0.05)
print("Run TSNE and UMAP")
tsne_obj <- RunTSNE(seurat_object, dims = 1:10, check_duplicates = FALSE)
umap_obj <- RunUMAP(seurat_object, dims = 1:10)
print("Plot TSNE and UMAP")
tsne_plot <- DimPlot(tsne_obj, reduction = "tsne") 
#tsne_plot + plot_annotation(title = 'DS1 tSNE')
umap_plot <- DimPlot(umap_obj, reduction = "umap") 
#umap_plot + plot_annotation(title = 'DS1 UMAP')

cluster_ids <- tsne_obj@meta.data$seurat_clusters
tsne1 <- tsne_obj@reductions$tsne@cell.embeddings[,1]
tsne2 <- tsne_obj@reductions$tsne@cell.embeddings[,2]
tsne_data <- data.frame(cluster_id = cluster_ids, x = tsne1, y = tsne2)
centroids <- aggregate(cbind(x, y) ~ cluster_id, data = tsne_data, FUN = mean)

numeric_cluster_ids <- as.numeric(as.character(tsne_data$cluster_id))
silhouette_scores <- silhouette(numeric_cluster_ids, dist(tsne_data[,c("x", "y")]))
sil_df <- data.frame(cluster = numeric_cluster_ids, 
                     silhouette_width = silhouette_scores[, "sil_width"])

avg_sil_width_by_cluster <- sil_df %>%
  group_by(cluster) %>%
  summarise(avg_silhouette_width = mean(silhouette_width))

print(avg_sil_width_by_cluster)

centroids <- aggregate(cbind(x, y) ~ cluster_id, data = tsne_data, FUN = mean)

wcss_per_cluster <- sapply(unique(tsne_data$cluster_id), function(cluster) {
  points_in_cluster <- tsne_data[tsne_data$cluster_id == cluster, c("x", "y")]
  centroid <- centroids[centroids$cluster_id == cluster, c("x", "y")]
  
  points_matrix <- as.matrix(points_in_cluster)
  centroid_matrix <- as.matrix(centroid[rep(1, nrow(points_in_cluster)), ])
  
  sum(rowSums((points_matrix - centroid_matrix)^2))
})

names(wcss_per_cluster) <- unique(tsne_data$cluster_id)
wcss_per_cluster <- wcss_per_cluster[order(as.numeric(names(wcss_per_cluster)))]
res_df <- data.frame(
  cluster_id = as.numeric(names(wcss_per_cluster)),
  avg_sil_width = avg_sil_width_by_cluster$avg_silhouette_width,
  WCSS = wcss_per_cluster
)
print(res_df)

sigmas <- sapply(unique(tsne_data$cluster_id), function(cluster) {
  points <- tsne_data[tsne_data$cluster_id == cluster, c("x", "y")]
  centroid <- centroids[centroids$cluster_id == cluster, c("x", "y")]
  
  points_mat <- as.matrix(points)
  centroid_mat <- as.matrix(centroid[rep(1, nrow(points)), ])
  
  d_squared <- rowSums((points_mat - centroid_mat)^2)
  
  mean(sqrt(d_squared))
})

c_dists <- as.matrix(dist(centroids[, c("x", "y")]))

db_index <- mean(sapply(1:length(unique(tsne_data$cluster_id)), function(i) {
  max((sigmas[i] + sigmas[-i]) / c_dists[i, -i])
}))

# Print the Davies-Bouldin Index
print(db_index)

tsne_plot + geom_point(data = centroids, aes(x = x, y = y), shape = 4, color = "black", size = 5) + ggtitle('DS3 tSNE')


