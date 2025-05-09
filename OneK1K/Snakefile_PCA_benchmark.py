import os
import pandas as pd


# adata = "/omics/groups/OE0540/internal_temp/users/LIVI/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K.h5ad" # Adata with X_pca at the single-cell level in obsm
data_dir = "/omics/groups/OE0540/internal/projects/LIVI"
adata = os.path.join(data_dir, "RNA_counts", "LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K_aggr-donor-celltype.h5ad") # Adata with GEX aggregated at the donor level within each celltype
genotype_dir = os.path.join(data_dir, "OneK1K", "filter_vcf_r08")
PLINK = os.path.join(genotype_dir, "ALL_chroms.dose.filtered.R2_0.8_001MAF_HWE_trans-eQTLGen")
KINSHIP = os.path.join(genotype_dir, "Kinship_square_rel_ALL_chroms.dose.filtered.R2_0.8_001MAF_HWE_pruned.tsv")

CELLTYPES = pd.read_table("/omics/groups/OE0540/internal/users/danai/Data/OneK1K/celltypes.txt",
                          names=["celltype"], index_col=False, sep="\t"
                         )["celltype"].str.strip().values.tolist()
CELLTYPES = [ct for ct in CELLTYPES if ct not in ["Platelets", "Erythrocytes"]]

# output_dir = os.path.join(genotype_dir, "Benchmarks", "PCA")
output_dir = os.path.join(data_dir, "Benchmarks", "PCA")
donor_col = "individual"
sex_col = "sex"
age_col = "age"
celltype_col = "cell_label"

N_PCS = [20, 50, 100, 300, 500, 700, 900]
output_dir = os.path.join(output_dir, "{n_pcs}_PCs")
FDR = 0.05
FDR_method = "Storey"


rule all:
    input:
        expand(os.path.join(output_dir, "{celltype}_{n_pcs}PCs_LMM_results_PCA-benchmark.tsv"), celltype=CELLTYPES, n_pcs=N_PCs),
        expand(os.path.join(output_dir, "{celltype}_{n_pcs}PCs_LMM_results_"+str(FDR_method)+"-"+str(FDR)+"_PCA-benchmark.tsv"), celltype=CELLTYPES, n_pcs=N_PCs),    

rule test_PCs_celltype:
    input:
        adata = adata,
    output:
        os.path.join(output_dir, "{celltype}_{n_pcs}PCs_LMM_results_PCA-benchmark.tsv"),
        os.path.join(output_dir, "{celltype}_{n_pcs}PCs_LMM_results_"+str(FDR_method)+"-"+str(FDR)+"_PCA-benchmark.tsv")
    params:
        output_dir = output_dir,
        plink = PLINK,
        kinship = KINSHIP,
       # n_pcs = n_pcs,
        celltype_column=celltype_col,
        individual_column=donor_col,
        sex_column=sex_col,
        age_column=age_col,
        fdr=FDR,
        fdr_method=FDR_method,
        ofp = "{celltype}_{n_pcs}PCs",
    conda:
        "LIVIenv"
    shell:
        """
        python /omics/groups/OE0540/internal_temp/users/danai/scripts/test_PCs_benchmark_aggregated-GEX.py --output_dir {params.output_dir} --adata {input.adata} --n_pcs {wildcards.n_pcs} --genotype_matrix {params.plink} --plink --kinship {params.kinship} --individual_column {params.individual_column} --celltype_col {params.celltype_column} --celltype {wildcards.celltype} --sex_column {params.sex_column} --age_column {params.age_column} --output_file_prefix {params.ofp} -fdr {params.fdr} --fdr_method {params.fdr_method}
        """
