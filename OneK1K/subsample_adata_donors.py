import os
import random
import numpy as np
import pandas as pd
import scanpy as sc


data_dir = "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts"
output_dir = os.path.join(data_dir, "Data_augmentation")

generator = np.random.default_rng(32)

n_cells = {}
n_pools = {}

adata = sc.read_h5ad(
    os.path.join(data_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K.h5ad")
)
n_cells["981"] = adata.obs.shape[0]
n_pools["981"] = adata.obs.pool.nunique()

donors700 = generator.choice(adata.obs.individual.unique(), size=700, replace=False)
adata_sub = adata[adata.obs.loc[adata.obs.individual.isin(donors700)].index, :].copy()
n_cells["700"] = adata_sub.obs.shape[0]
n_pools["700"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_700donors.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_700donors_adata_obs.tsv"),
    sep="\t", index=True, header=True
)

donors500 = generator.choice(adata.obs.individual.unique(), size=500, replace=False)
adata_sub = adata[adata.obs.loc[adata.obs.individual.isin(donors500)].index, :].copy()
n_cells["500"] = adata_sub.obs.shape[0]
n_pools["500"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_500donors.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_500donors_adata_obs.tsv"), 
    sep="\t", index=True, header=True
)

donors300 = generator.choice(adata.obs.individual.unique(), size=300, replace=False)
adata_sub = adata[adata.obs.loc[adata.obs.individual.isin(donors300)].index, :].copy()
n_cells["300"] = adata_sub.obs.shape[0]
n_pools["300"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_300donors.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_300donors_adata_obs.tsv"), 
    sep="\t", index=True, header=True
)

donors100 = generator.choice(adata.obs.individual.unique(), size=100, replace=False)
adata_sub = adata[adata.obs.loc[adata.obs.individual.isin(donors100)].index, :].copy()
n_cells["100"] = adata_sub.obs.shape[0]
n_pools["100"] = adata_sub.obs.pool.nunique()
adata_sub.write(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_100donors.h5ad")
)
adata_sub.obs.to_csv(
    os.path.join(output_dir, "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_100donors_adata_obs.tsv"),
    sep="\t", index=True, header=True
)

info_df = pd.DataFrame.from_dict(n_cells, orient="index", columns=["n_cells"])
info_df = pd.concat([info_df, pd.DataFrame.from_dict(n_pools, orient="index", columns=["n_pools"])],
                    axis=1, ignore_index=False)
info_df.to_csv(os.path.join(output_dir, "subsampled_donors_meta.tsv"), sep="\t", index=True, header=True)
