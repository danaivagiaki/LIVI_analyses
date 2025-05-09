import argparse
import os
import re
import sys
import warnings
from anndata import AnnData
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from glimix_core.lmm import LMM
from multipy.fdr import qvalue
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from pandas_plink import read_plink
from scipy.stats import chi2, norm, probplot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, quantile_transform


def QQplot(pvalues: Union[list, np.ndarray], savefig: Optional[str] = None) -> None:
    """Generate a QQ plot to assess the deviation of observed p-values from expected uniform
    distribution.

    Parameters
    ----------
        pvalues (array-like): The observed p-values.
        savefig (str or None): If provided, the path to save the generated QQ plot. Default is None.

    Returns
    -------
        None
    """
    (osm, osr), _ = probplot(pvalues, dist="uniform")
    b, a = np.polyfit(osm, osr, deg=1)
    ax = plt.gca()
    df = pd.DataFrame({"osm": -np.log10(osm), "osr": -np.log10(osr)})
    sns.scatterplot(x="osm", y="osr", data=df, ax=ax, edgecolor=None)
    x = np.linspace(0, ax.get_xlim()[1])
    ax.plot(x, a + b * x, c="lightgrey", linestyle=":", scaley=True)
    ax.set(xlabel=r"Expected $-\log_{10} P$", ylabel=r"Observed $-\log_{10} P$")

    if savefig is not None:
        plt.savefig(savefig, transparent=True, dpi=200, bbox_inches="tight")



def lrt_pvalues(null_lml: float, alt_lmls: Union[float, np.ndarray], dof: int = 1) -> np.ndarray:
    """Compute p-values from likelihood ratios.

    Parameters
    ----------
    null_lml (float): Log-likelihood of the null model.
    alt_lmls (Union[float, np.ndarray]): Log-likelihoods of the alternative models.
    dof (Optional[int]): Degrees of freedom for the chi-squared distribution. Default is 1.

    Returns
    -------
    np.ndarray: Likelihood ratio test p-values.
    """
    import numpy as np

    super_tiny = np.finfo(float).tiny
    tiny = np.finfo(float).eps

    lrs = np.clip(-2 * null_lml + 2 * np.asarray(alt_lmls, float), super_tiny, np.inf)
    pv = chi2(df=dof).sf(lrs)

    return np.clip(pv, super_tiny, 1 - tiny)


def setup_covars(adata: AnnData, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if "X_pca" not in adata.obsm.keys():
        n_pcs = min(adata.shape[0], adata.shape[1])
        n_pcs = min(n_pcs, args.n_pcs) -1
        sc.tl.pca(adata, n_comps=n_pcs, zero_center=True, svd_solver='arpack', random_state=32, mask_var=None)
    else:
        n_pcs = args.n_pcs

    PCs = pd.DataFrame(adata.obsm["X_pca"][:,:n_pcs], index=adata.obs.index, columns=[f"PC{i+1}" for i in range(n_pcs)])

    if args.celltype is not None:
        if args.batch_column is not None:
            covariates = adata.obs.filter([args.individual_column, args.batch_column, args.celltype_column]).drop_duplicates()
            covariates = covariates.assign(
                sample_id = covariates[args.celltype_column].astype(str) + "__" + covariates[args.individual_column].astype(str)+"__"+covariates[args.batch_column].astype(str)
            ) 
        else:
            covariates = adata.obs.filter([args.individual_column, args.celltype_column]).drop_duplicates()
            covariates = covariates.assign(
                sample_id = covariates[args.celltype_column].astype(str) + "__" + covariates[args.individual_column].astype(str)
            ) 
        covariates = covariates.drop(columns=[args.celltype_column]).set_index("sample_id")     
    else:
        if args.batch_column is not None:
            covariates = adata.obs.filter([args.individual_column, args.batch_column]).drop_duplicates()
            covariates = covariates.assign(
                sample_id = covariates[args.individual_column].astype(str)+"__"+covariates[args.batch_column].astype(str)
            ) 
        else:
            covariates = adata.obs.filter([args.individual_column]).drop_duplicates()
            covariates = covariates.assign(
                sample_id = covariates[args.individual_column]
            ) 

    if args.sex_column is not None:
        adata.obs[args.sex_column] = pd.Categorical(adata.obs[args.sex_column])
        adata.obs[args.sex_column] = adata.obs[args.sex_column].cat.rename_categories(
            {
                adata.obs[args.sex_column].cat.categories[0]: 1,
                adata.obs[args.sex_column].cat.categories[1]: 0,
            },
        )
        tmp = adata.obs.filter([args.celltype_column, args.individual_column, args.batch_column, args.sex_column]).drop_duplicates()
        if args.celltype is not None:
            if args.batch_column is not None:
                tmp = tmp.assign(sample_id = tmp[args.celltype_column].astype(str)+"__"+tmp[args.individual_column].astype(str) + "__"+ tmp[args.batch_column].astype(str))
            else:
                tmp = tmp.assign(sample_id = tmp[args.celltype_column].astype(str)+"__"+tmp[args.individual_column].astype(str))
        else:
            if args.batch_column is not None:
                tmp = tmp.assign(sample_id = tmp[args.individual_column].astype(str) + "__"+ tmp[args.batch_column].astype(str))
            else:
                tmp = tmp.assign(sample_id = tmp[args.individual_column].astype(str))
        covariates = covariates.merge(
            tmp.set_index("sample_id").filter([args.sex_column]),
            right_index=True,
            left_index=True,
            how="left"
        )
    if args.age_column is not None:
        tmp = adata.obs.filter([args.celltype_column, args.individual_column, args.batch_column, args.age_column]).drop_duplicates()
        tmp = tmp.assign(
            age_scaled = StandardScaler().fit_transform(
                tmp[args.age_column].to_numpy().reshape(-1, 1)
            )
        )
        if args.celltype is not None:
            if args.batch_column is not None:
                tmp = tmp.assign(sample_id = tmp[args.celltype_column].astype(str)+"__"+tmp[args.individual_column].astype(str) + "__" + tmp[args.batch_column].astype(str))
            else:
                tmp = tmp.assign(sample_id = tmp[args.celltype_column].astype(str)+"__"+tmp[args.individual_column].astype(str))
        else:
            if args.batch_column is not None:
                tmp = tmp.assign(sample_id = tmp[args.individual_column].astype(str) + "__" + tmp[args.batch_column].astype(str))
            else:
                tmp = tmp.assign(sample_id = tmp[args.individual_column].astype(str))
        covariates = covariates.merge(
            tmp.set_index("sample_id").filter(["age_scaled"]),
            right_index=True,
            left_index=True,
            how="left"
        )

    return PCs, covariates
        

def LMM_test_feature(
    feature_id: Union[int, str],
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    G: pd.DataFrame,
    QS: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray],
    quantile_norm: bool,
) -> pd.DataFrame:
    """Perform a linear mixed model (LMM) test for the effect of a genetic variable (e.g. SNP or
    PRS) on a specified phenotype feature (e.g. a gene or factor).

    Parameters
    ----------
    feature_id (Union[int, str]): Identifier for the specific phenotype feature.
    phenotype_df (pd.DataFrame): DataFrame containing the values of all phenotype features.
    covariates_df (pd.DataFrame): DataFrame containing sample covariates to be included
        as fixed effects in the LMM.
    G (pd.DataFrame): DataFrame containing the genetic data.
    QS (tuple): Economic eigendecomposition in the form of ((Q0, Q1), S0) of a kinship
        matrix K (used as covariance of the random genetic effect).
    quantile_norm (bool): Flag indicating whether quantile normalization should be
        applied to the phenotype.

    Returns
    -------
    pd.DataFrame: DataFrame containing results of the LMM test, including phenotype feature ID,
        genetic variable ID, effect size, effect size standard error, and p-value.
    """
    feature_phenotype = phenotype_df.loc[feature_id]
    # Remove samples where the feature is NaN
    feature_phenotype = feature_phenotype.dropna()

    if covariates_df.empty:
        covariates_matrix = np.ones((feature_phenotype.shape[0], 1))
    else:
        covariates_df["intercept"] = 1.0
        covariates_matrix = covariates_df.to_numpy().astype(np.float32)

    G_matrix = G.values  # individuals x G_variable

    if quantile_norm:
        phenotype = quantile_transform(
            feature_phenotype.values.reshape(-1, 1), output_distribution="normal"
        )
    else:
        phenotype = feature_phenotype

    null_lmm = LMM(phenotype, covariates_matrix, QS, restricted=False)
    null_lmm.fit(verbose=False)
    # Log of the marginal likelihood of null model:
    null_loglk = null_lmm.lml()

    # Test the effect of G_variable
    flmm = null_lmm.get_fast_scanner()

    scannerOut = flmm.fast_scan(G_matrix, verbose=False)
    alt_loglks = scannerOut["lml"]
    effsizes = np.asarray(scannerOut["effsizes1"])
    effsizes_se = np.asarray(scannerOut["effsizes1_se"])
    # Compute p-values from likelihood ratios: null model vs. full model with genetics
    pvalues = np.asarray(lrt_pvalues(null_loglk, alt_loglks))

    feature_results = pd.DataFrame(
        {
            "feature_id": [feature_id] * G.shape[1],
            "variable": G.columns,
            "effect_size": effsizes,
            "effect_size_se": effsizes_se,
            "p_value": pvalues,
        }
    )

    return feature_results


def FDR_correction(testing_results: pd.DataFrame, cut_off: float = 0.05, method: str = "Storey") -> pd.DataFrame:
    """Perform False Discovery Rate (FDR) correction on testing results.

    Parameters
    ----------
    testing_results (pd.DataFrame): DataFrame containing testing results.
    cut_off (float, optional): False discovery rate (FDR) threshold for significance.
        Default is 0.05.
    method (str, optional): Method used for adjustment of pvalues. Supported are FDR controlling methods, 
        specifically Benjamini-Hochberg, Benjamini-Yekutieli and Storey's Q-value method.

    Returns
    -------
    pd.DataFrame: DataFrame containing testing results after FDR correction.
    """

    ## Multiple testing correction across everything
    if method in ["Storey", "storey", "qvalue", "q_value"]:
        testing_results = testing_results.assign(
            corrected_pvalue=qvalue(testing_results["p_value"].to_numpy(), threshold=cut_off)[1]
        )
    elif method in ["Benjamini-Hochberg", "benjamini_hochberg", "BH", "bh", "fdr_bh"]:
        testing_results = testing_results.assign(
            corrected_pvalue= multitest.multipletests(testing_results["p_value"].values, method="fdr_bh", is_sorted=False, returnsorted=False)[1]
        )
    elif method in ["Benjamini-Yekutieli", "benjamini_yekutieli", "BY", "by", "fdr_by"]:
        testing_results = testing_results.assign(
            corrected_pvalue= multitest.multipletests(testing_results["p_value"].values, method="fdr_by", is_sorted=False, returnsorted=False)[1]
        )
    else:
        raise ValueError(f"Unsupported method {method}. Valid options are 'Storey', 'Benjamini-Hochberg' and 'Benjamini-Yekutieli'.")


    testing_results_sign = testing_results.loc[testing_results.corrected_pvalue < cut_off]
    print(f"number of fQTLs: {testing_results_sign.shape[0]}")
    print(f"number of unique fSNPs: {testing_results_sign.SNP_id.nunique()}")
    print(f"number of unique factors: {testing_results_sign.Factor.nunique()}")

    return testing_results_sign


def run_LIVI_genetic_association_testing(
    PCs,
    GT_matrix,
    individual_column,
    output_dir,
    output_file_prefix,
    fdr_method="BH",
    Kinship=None,
    bim=None,
    covariates=None,
    quantile_norm=True,
    fdr_threshold=None,
    return_associations=False,
):
    if covariates is not None:
        GT_matrix = GT_matrix.loc[covariates[individual_column]]

    if Kinship is not None:
        kinship = Kinship
        kinship_mat = (
            kinship.loc[covariates[individual_column], covariates[individual_column]].to_numpy()
            if covariates is not None
            else kinship.to_numpy()
        )
    else:
        kinship_mat = np.dot(GT_matrix.to_numpy(), GT_matrix.T.to_numpy())
        kinship_mat = normalise_covariance(kinship_mat)
    QS = economic_qs(kinship_mat)

    # Remove individual column from covariates
    covariates = covariates.drop(columns=[individual_column])
    
    results = pd.DataFrame(
        columns=["Factor", "SNP_id", "effect_size", "effect_size_se", "p_value"]
    )

    for f in PCs.columns:
        print(f"Testing: {f}")
        results_factor = LMM_test_feature(
            feature_id=f,
            phenotype_df=PCs.T,
            covariates_df=covariates,
            G=GT_matrix,
            QS=QS,
            quantile_norm=quantile_norm,
        )
        results_factor.rename(
            columns={"feature_id": "Factor", "variable": "SNP_id"}, inplace=True
        )
        
        if results.empty:
            results = results_factor
        else:
            results = pd.concat([results, results_factor], axis=0)
    if bim is not None:
        results = results.merge(
            bim.filter(["snp", "a1"]).rename(
                columns={"snp": "SNP_id", "a1": "assessed_allele"}
            ),
            on="SNP_id",
            how="left",
        )

    filename = (
        f"{output_file_prefix}_LMM_results_PCA-benchmark.tsv"
        if output_file_prefix is not None 
        else "LMM_results_PCA-benchmark.tsv"
    )
    results.to_csv(os.path.join(output_dir, filename), sep="\t", header=True, index=False)

    qqplot_filename = (
        f"{output_file_prefix}_QQplot_LMM_results_PCA-benchmark.png"
        if output_file_prefix is not None 
        else "QQplot_LMM_results_PCA-benchmark.png"
    )
    QQplot(
        results.p_value,
        savefig=os.path.join(output_dir, qqplot_filename),
    )
    plt.close()

    histplot_filename = f"{output_file_prefix}_Pvalue-Histogram_PCA-benchmark.png"
    sns.histplot(results.p_value, bins=500, color="royalblue")
    plt.savefig(os.path.join(output_dir, histplot_filename), transparent=True, bbox_inches="tight", dpi=200)

    threshold = fdr_threshold if fdr_threshold is not None else 0.05
    results_sign = FDR_correction(results, cut_off=threshold, method=fdr_method)

    filename_sign = (
        f"{output_file_prefix}_LMM_results_{fdr_method}-{threshold}_PCA-benchmark.tsv"
        if output_file_prefix
        else f"LMM_results_{fdr_method}-{threshold}_PCA-benchmark.tsv"
    )
    results_sign.to_csv(
        os.path.join(output_dir, filename_sign), sep="\t", header=True, index=False
    )


def validate_and_read_passed_args(
    args: argparse.Namespace,
) -> Tuple[
    str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Validate the passed arguments and read the corresponding files.

    Parameters
    ----------
    args (argparse.Namespace): Parsed command-line arguments.
    
    Returns
    -------
    Tuple[str, str, AnnData, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        output_dir (str): Output directory to save the testing results.
        of_prefix (str): Output file prefix.
        metadata (pd.DataFrame): DataFrame containing cell metadata.
        GT_matrix(pd.DataFrame): Genotype matrix (donors x SNPs).
        bim (pd.DataFrame): SNP information contained in the .bim file, if PLINK genotype matrix is used, otherwise None.
        kinship (pd.DataFrame): Kinship matrix if provided, otherwise None.
        U_context (pd.DataFrame): LIVI cell-state-specific genetic embedding.
        V_persistent (pd.DataFrame): LIVI persistent genetic embedding if applicable, otherwise None.
    """ 
    
    assert os.path.isfile(args.adata), "AnnData file not found."

    output_dir = args.output_dir
    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)

    of_prefix = (
        args.output_file_prefix
        if args.output_file_prefix
        else os.path.basename(args.adata)
    )

    adata = sc.read_h5ad(args.adata)

    assert (
        args.individual_column in adata.obs.columns
    ), f"`individual_column`: '{args.individual_column}' not in cell metadata columns."
    if args.celltype_column:
        assert (
            args.celltype_column in adata.obs.columns
        ), f"`celltype_column`: '{args.celltype_column}' not in adata.obs columns."
    if args.batch_column:
        assert (
            args.batch_column in adata.obs.columns
        ), f"`batch_column`: '{args.batch_column}' not in adata.obs columns."
    if args.sex_column:
        assert (
            args.sex_column in adata.obs.columns
        ), f"`sex_column`: '{args.sex_column}' not in adata.obs columns."
    if args.age_column:
        assert (
            args.age_column in adata.obs.columns
        ), f"`age_column`: '{args.age_column}' not in adata.obs columns."
 
    if args.celltype:
        assert args.celltype_column is not None, "`celltype_column` argument is required, if testing only for a specific cell type"
        celltype = args.celltype.replace("-", " ")
        if celltype not in adata.obs[args.celltype_column].unique():
            raise ValueError(f"No cells belonging to cell type {celltype}")
        else:
            adata = adata[adata.obs.loc[adata.obs[args.celltype_column] == celltype].index,:]
    
    if args.kinship is not None:
        assert os.path.isfile(args.kinship), "Kinship matrix file not found."
        _, ext = os.path.splitext(args.kinship)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Kinship matrix must be either .tsv or .csv. File format provided: {ext}."
            )
        kinship = pd.read_csv(args.kinship, index_col=0, sep="\t" if ext == ".tsv" else ",")
        kinship = kinship.loc[
            adata.obs[args.individual_column].unique(), adata.obs[args.individual_column].unique()
        ]
        if kinship.shape[0] == 0:
            raise ValueError(
                "Individual IDs in cell metadata do not match individual IDs in the kinship matrix."
            )
    else:
        kinship = None

    if args.plink:
        bim, fam, bed = read_plink(args.genotype_matrix, verbose=False)
        GT_matrix = pd.DataFrame(
            bed.compute(), index=bim.snp, columns=fam.iid
        )  # SNPs x donors
    else:
        assert os.path.isfile(args.genotype_matrix), "Genotype matrix file not found."
        _, ext = os.path.splitext(args.genotype_matrix)
        if ext not in [".tsv", ".csv"]:
            raise TypeError(
                f"Genotype matrix must be either .tsv or .csv. File format provided: {ext}. To use a PLINK matrix please use the --plink flag"
            )
        GT_matrix = pd.read_csv(
            args.genotype_matrix, index_col=0, sep="\t" if ext == ".tsv" else ","
        )
        bim = None

    GT_matrix = GT_matrix.filter(adata.obs[args.individual_column].unique())
    if GT_matrix.shape[1] == 0:
        raise ValueError(
            "Individual IDs in cell metadata do not match individual IDs in the genotype matrix."
        )
    GT_matrix = GT_matrix.T  # donors x SNPs

    return (
        output_dir,
        of_prefix,
        adata,
        GT_matrix,
        bim,
        kinship,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        required=True,
        help="Absolute path of the directory to save the testing results.",
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Absolute path of the AnnData file containing the scRNA-seq data.",
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        required=True,
        help="Number of epxression principal components to compute and test.",
    )
    parser.add_argument(
        "--individual_column",
        "-id",
        type=str,
        required=True,
        help="Column name in cell metadata (adata.obs) indicating the individual the sample (cell) comes from.",
    )
    parser.add_argument(
        "--genotype_matrix",
        "-GT_matrix",
        type=str,
        required=True,
        help="Absolute path of the .tsv file with the genotype matrix (the SNPs to test against LIVI's individual embeddings). For PLINK files please use in addition the --plink flag.",
    )
    parser.add_argument(
        "--plink",
        action="store_true",
        default=False,
        help="If PLINK genotype files (bim, bed, fam if `method` is LIMIX, or pgen, pvar, psam if `method` is tensorQTL) are provided instead of a GT matrix in .tsv format.",
    )
    parser.add_argument(
        "--kinship",
        "-K",
        type=str,
        help="Absolute path of the .tsv file with the Kinship matrix (e.g. generated with PLINK) to be used for relatedness/population structure correction during variant testing. Required when testing_method == 'LIMIX'",
    )
    parser.add_argument(
        "--celltype_column",
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the cell type.",
    )
    parser.add_argument(
        "--celltype",
        type=str,
        help="Run testing only for this cell type.",
    )
    parser.add_argument(
        "--batch_column",
        default=None,
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the experimental batch the sample (cell) comes from.",
    )
    parser.add_argument(
        "--sex_column",
        default=None,
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the sex of the individual.",
    )
    parser.add_argument(
        "--age_column",
        default=None,
        type=str,
        help="Column name in cell metadata (adata.obs) indicating the age of the individual.",
    )
    parser.add_argument(
        "--quantile_normalise",
        action="store_true",
        default=False,
        help="Whether to quantile normalise LIVI's individual embeddings prior to variant association testing.",
    )
    parser.add_argument(
        "--fdr_threshold",
        "-fdr",
        default=None,
        type=float,
        help="False discovery rate (FDR) threshold for multiple testing correction.",
    )
    parser.add_argument(
        "--fdr_method",
        default=None,
        type=str,
        choices=["Storey", "qvalue", "Benjamini-Hochberg", "BH", "Benjamini-Yekutieli", "BY"],
        help="False discovery rate (FDR) controlling method for multiple testing correction.",
    )
    parser.add_argument(
        "--output_file_prefix",
        "-ofp",
        default=None,
        type=str,
        help="Common prefix of the output results files.",
    )

    args = parser.parse_args()

    (
        od,
        of_prefix,
        adata,
        GT_matrix,
        bim,
        kinship,
    ) = validate_and_read_passed_args(args)

    PCs, covariates = setup_covars(adata, args)

    run_LIVI_genetic_association_testing(
        PCs=PCs,
        GT_matrix=GT_matrix,
        individual_column=args.individual_column,
        fdr_method=args.fdr_method,
        bim=bim,
        Kinship=kinship,
        output_dir=od,
        output_file_prefix=of_prefix,
        covariates=covariates,
        quantile_norm=args.quantile_normalise,
        fdr_threshold=(
            args.fdr_threshold if args.fdr_threshold else None
        ),
    )



