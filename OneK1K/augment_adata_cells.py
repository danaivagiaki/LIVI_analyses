import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

data_dir = "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts"
output_dir = os.path.join(data_dir, "Data_augmentation")

adata = sc.read_h5ad(
    os.path.join(data_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K.h5ad")
)

# donors = ["782_783", "116_116"] # These donors belong to different latent, but both have n_cells == median(n_cells_per_donor) (that is 1,181 cells)
# adata_782_783 = adata[adata.obs.loc[adata.obs.individual == "782_783"].index, :].copy()
# adata_116_116 = adata[adata.obs.loc[adata.obs.individual == "116_116"].index, :].copy()

n_cells_per_donor = adata.obs.value_counts("individual")
median_cells = n_cells_per_donor.median()
# Select donors with n_cells around the median(n_cells_per_donor)
donors_around_the_median = n_cells_per_donor[(median_cells - 100 < n_cells_per_donor) & (n_cells_per_donor < median_cells + 100)].index

adata_sub = adata[adata.obs.loc[adata.obs.individual.isin(donors_around_the_median)].index,:].copy()

adata_aug1 = anndata.concat(adatas=[adata, adata_sub],
                            axis=0, 
                            join="inner",
                            merge="same", 
                            uns_merge=None,
                            label=None,
                            keys=None, 
                            index_unique=None,
                            fill_value=None, 
                            pairwise=False)

print(adata_aug1.shape) # 1,504,589 cells
adata_aug1.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_1.5M.h5ad")
)
adata_aug1.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_1.5M_adata_obs.tsv"),
    sep="\t", index=True, header=True
)

adata_aug2 = anndata.concat(adatas=[adata_aug1, adata_sub],
                            axis=0, 
                            join="inner",
                            merge="same", 
                            uns_merge=None,
                            label=None,
                            keys=None, 
                            index_unique=None,
                            fill_value=None, 
                            pairwise=False)
print(adata_aug2.shape) # 1,836,387 cells
adata_aug2.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_1.8M.h5ad")
)
adata_aug2.obs.to_csv(
    os.path.join(output_dir,"LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_1.8M_adata_obs.tsv"), 
    sep="\t", index=True, header=True
)

adata_aug3 = anndata.concat(adatas=[adata_aug2, adata_sub],
                            axis=0, 
                            join="inner",
                            merge="same", 
                            uns_merge=None,
                            label=None,
                            keys=None, 
                            index_unique=None,
                            fill_value=None, 
                            pairwise=False)
print(adata_aug3.shape) # 2,168,185 cells
adata_aug3.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_2M.h5ad")
)
adata_aug3.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_2M_adata_obs.tsv"),
    sep="\t", index=True, header=True
)

