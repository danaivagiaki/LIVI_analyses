import os
import random
import numpy as np
import pandas as pd
import scanpy as sc

n_donors = {}
n_pools = {}

data_dir = "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts"
output_dir = os.path.join(data_dir, "Data_augmentation")

adata = sc.read_h5ad(
    os.path.join(data_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K.h5ad")
)
n_donors["1.18M"] = adata.obs.individual.nunique()
n_pools["1.18M"] = adata.obs.pool.nunique()

adata_sub = sc.pp.subsample(adata, n_obs=700000, random_state=32, copy=True)
n_donors["700K"] = adata_sub.obs.individual.nunique()
n_pools["700K"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_700K.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_700K_adata_obs.tsv"), 
    sep="\t", index=True, header=True
)

adata_sub = sc.pp.subsample(adata, n_obs=500000, random_state=32, copy=True)
n_donors["500K"] = adata_sub.obs.individual.nunique()
n_pools["500K"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_500K.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_500K_adata_obs.tsv"), 
    sep="\t", index=True, header=True
)

adata_sub = sc.pp.subsample(adata, n_obs=300000, random_state=32, copy=True)
n_donors["300K"] = adata_sub.obs.individual.nunique()
n_pools["300K"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_300K.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_300K_adata_obs.tsv"),
    sep="\t", index=True, header=True
)

adata_sub = sc.pp.subsample(adata, n_obs=100000, random_state=32, copy=True)
n_donors["100K"] = adata_sub.obs.individual.nunique()
n_pools["100K"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_100K.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_100K_adata_obs.tsv"),
    sep="\t", index=True, header=True
)

info_df = pd.DataFrame.from_dict(n_donors, orient="index", columns=["n_donors"])
info_df = pd.concat([info_df, pd.DataFrame.from_dict(n_pools, orient="index", columns=["n_pools"])],
                    axis=1, ignore_index=False)
info_df.to_csv(os.path.join(output_dir, "subsampled_cells_meta.tsv"), sep="\t", index=True, header=True)
