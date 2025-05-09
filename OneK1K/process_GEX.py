import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

adata = sc.read_h5ad("/omics/groups/OE0540/internal/projects/DeepCellRegMap/OneK1K/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth.h5ad")

# Remove non-immune cells
adata = adata[adata.obs.loc[~adata.obs.major_celltype.isin(["Platelet", "Eryth"])].index,:]

adata.write("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells.h5ad")

adata.obs.to_csv("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_adata-obs.tsv", 
                 sep="\t", index=True, header=True)
adata.var.to_csv("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_adata-var.tsv", 
                 sep="\t", index=True, header=True)

# Select top 10K genes
sc.pp.highly_variable_genes(adata, n_top_genes=10000, flavor="seurat")

# Additionally select top 10K highest expressed genes (adapted from `sc.pl.highest_expr_genes`)
n_top = 10000
norm_dict = sc.pp.normalize_total(adata, target_sum=100, inplace=False)
if issparse(norm_dict["X"]):
    mean_percent = norm_dict["X"].mean(axis=0).A1
    top_idx = np.argsort(mean_percent)[::-1][:n_top]
    counts_top_genes = norm_dict["X"][:, top_idx].toarray()
else:
    mean_percent = norm_dict["X"].mean(axis=0)
    top_idx = np.argsort(mean_percent)[::-1][:n_top]
    counts_top_genes = norm_dict["X"][:, top_idx]
top_genes = adata.var_names[top_idx]

HVG_HEX = list(set(adata.var[adata.var.highly_variable].index) | set(top_genes))
print(f"{len(HVG_HEX)} HVG/HEX genes selected in total.") # Finally 8,746 genes

adata = adata[:,HVG_HEX]

adata.write("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K.h5ad")

adata.obs.to_csv("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_adata-obs.tsv", 
                 sep="\t", index=True, header=True)
adata.var.to_csv("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_adata-var.tsv", 
                 sep="\t", index=True, header=True)

