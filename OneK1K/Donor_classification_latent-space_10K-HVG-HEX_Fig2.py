# %%
import sys
import os
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("/omics/groups/OE0540/internal/users/danai/scripts/LIVI/")
from src.data_modules.livi_data import LIVIDataModule
from src.models.livi_experimental import LIVI_cis_gen
from src.models.components.mlp import create_mlp

# %%
pl.seed_everything(32)

# %%
results_dir = "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/LIVI2_testing_results"
model = "2024-12-05_10-20_LIVIcis-cell-state_zdim15_700-gxc_5-persistent_60-0-warm-up_no-adversary_Gseed200_hvg-hex-10K_larger-encoder_best"
model_results_dir = os.path.join(results_dir, model)
os.path.isdir(model_results_dir)

# %%
ckpt_dir = os.path.join("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/LIVI_checkpoints", model.replace("_best", ""), "checkpoints")
os.listdir(ckpt_dir)

# %%
myLIVI = LIVI_cis_gen.load_from_checkpoint(os.path.join(ckpt_dir, "LIVI2_onek1k_all-protein-genes_epoch=0317_hp_metric=2857.55347.ckpt"),
                                           map_location=torch.device("cpu"))

adata = sc.read_h5ad("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/RNA_counts/LogNorm_counts_across_celltypes_DCRM_protein-coding_Azimuth_only-immune-cells_HVG-HEX-10K.h5ad")

# %%
Y, _ = pd.factorize(adata.obs["individual"])
Y = torch.from_numpy(Y).to(torch.long)

# %%
zbase = pd.read_csv(os.path.join(model_results_dir,
                                    f"{model}_cell-state_latent.tsv"),
                               sep="\t", index_col=0)
zbase = zbase.loc[adata.obs.index]
# %%
U = pd.read_csv(os.path.join(model_results_dir,
                                    f"{model}_U_embedding.tsv"),
                               sep="\t", index_col=0)
U = U.loc[adata.obs["individual"]] # expanded to cell-level
U = torch.from_numpy(U.to_numpy()).to(torch.double)

# %%
V = pd.read_csv(os.path.join(model_results_dir,
                                    f"{model}_V_embedding.tsv"),
                               sep="\t", index_col=0)
V = V.loc[adata.obs["individual"]] # expanded to cell-level
V = torch.from_numpy(V.to_numpy()).to(torch.double)


# %%
"""
### Classify IID based on cell-state factors
"""

# %%
Classifier = create_mlp(
    input_size=myLIVI.hparams.z_dim,
    output_size=myLIVI.hparams.y_dim,
    hidden_dims=[100, 1000, 5000],
    layer_norm=False,
    device="cuda:0",
)
Classifier

# %%
device="cuda:0"
batch_size=1024
train_epochs=50
optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-4, weight_decay=0.0)

# %%
generator = torch.Generator().manual_seed(32)
zbase = torch.from_numpy(zbase.to_numpy()).to(torch.double)

dataset = TensorDataset(zbase, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
torch.save(test_loader, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/test_loader_cell-state.pt")


# %%
train_error = {}
validation_error = {}
best_epoch = 1
best_loss = 10e10

for epoch in range(1, train_epochs+1):
    batch_ce = 0.0

    for x, y in train_loader:
        # Pass through VAE
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

        ce_loss = F.cross_entropy(Classifier(x), y)
        batch_ce += ce_loss.item()

        # Backprop and optimize
        optimizer.zero_grad()
        try:
            ce_loss.backward()
        #   nn.utils.clip_grad_norm_(model.parameters(), 10000)
        except RuntimeError:
            print("Epoch: {}   batch: {}".format(epoch, b))

        optimizer.step()

    # Mean epoch loss over batches
    epoch_loss = batch_ce / len(train_loader)
    train_error[f"Epoch_{epoch}"] = epoch_loss

    # Calculate validation loss for epoch
    val_loss = 0.0
    with torch.no_grad():
        Classifier.eval()
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            ce_loss_val = F.cross_entropy(Classifier(x), y)
            val_loss += ce_loss_val.item()

    # Mean validation loss for the epoch over batches
    val_loss = val_loss / len(val_loader)
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}   Current Train Loss: {epoch_loss:.5f}, Current Val Loss: {val_loss:.5f}, Current best epoch: {best_epoch}"
        )
    validation_error[f"Epoch_{epoch}"] = val_loss

    if val_loss <= best_loss:
        best_loss = val_loss
        best_epoch = epoch

        torch.save(Classifier, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K//Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state_best-epoch.pt")

    torch.save(Classifier, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state.pt")


# %%
Classifier = torch.load("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state_best-epoch.pt")


# %%
test_loss_cell_state = 0.0
test_accuracy_cell_state = 0.0

with torch.no_grad():
    Classifier.eval()
    for x, y in test_loader:
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        y_hat = Classifier(x)
        ce_loss = F.cross_entropy(y_hat, y, reduction="mean")
        test_loss_cell_state += ce_loss
        # Get labels
        y_hat_labels = y_hat.softmax(dim=1).max(dim=1)[1]
        acc = accuracy_score(y.cpu().numpy(), y_hat_labels.cpu().numpy(), normalize=True)
        test_accuracy_cell_state += acc
 
test_loss_cell_state = test_loss_cell_state / len(test_loader)
test_loss_cell_state = test_loss_cell_state.cpu().numpy()
test_accuracy_cell_state = test_accuracy_cell_state/len(test_loader)

print(f"Test loss for IID based on cell-state latent: {test_loss_cell_state}")
print(f"Test accuracy for IID based on cell-state latent: {test_accuracy_cell_state}")


"""
### Classify IID based on U embedding
"""

# %%
Classifier = create_mlp(
    input_size=myLIVI.hparams.n_gxc_factors,
    output_size=myLIVI.hparams.y_dim,
    hidden_dims=[1000, 5000, 2000],
    layer_norm=False,
    device="cuda:0",
)
Classifier

# %%
device="cuda:0"
batch_size=1024
train_epochs=50
optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-4, weight_decay=0.0)

# %%
generator = torch.Generator().manual_seed(32)

dataset = TensorDataset(U, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
torch.save(test_loader, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/test_loader_U.pt")


# %%
train_error = {}
validation_error = {}
best_epoch = 1
best_loss = 10e10

for epoch in range(1, train_epochs+1):
    batch_ce = 0.0

    for x, y in train_loader:
        # Pass through VAE
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

        ce_loss = F.cross_entropy(Classifier(x), y)
        batch_ce += ce_loss.item()

        # Backprop and optimize
        optimizer.zero_grad()
        try:
            ce_loss.backward()
        #   nn.utils.clip_grad_norm_(model.parameters(), 10000)
        except RuntimeError:
            print("Epoch: {}   batch: {}".format(epoch, b))

        optimizer.step()

    # Mean epoch loss over batches
    epoch_loss = batch_ce / len(train_loader)
    train_error[f"Epoch_{epoch}"] = epoch_loss

    # Calculate validation loss for epoch
    val_loss = 0.0
    with torch.no_grad():
        Classifier.eval()
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            ce_loss_val = F.cross_entropy(Classifier(x), y)
            val_loss += ce_loss_val.item()

    # Mean validation loss for the epoch over batches
    val_loss = val_loss / len(val_loader)
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}   Current Train Loss: {epoch_loss:.5f}, Current Val Loss: {val_loss:.5f}, Current best epoch: {best_epoch}"
        )
    validation_error[f"Epoch_{epoch}"] = val_loss

    if val_loss <= best_loss:
        best_loss = val_loss
        best_epoch = epoch

        torch.save(Classifier, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-U-embedding_best-epoch.pt")

    torch.save(Classifier, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-U-embedding.pt")


# %%
Classifier = torch.load("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-U-embedding_best-epoch.pt")


# %%
test_loss_U = 0.0
test_accuracy_U = 0.0

with torch.no_grad():
    Classifier.eval()
    for x, y in test_loader:
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        y_hat = Classifier(x)
        ce_loss = F.cross_entropy(y_hat, y, reduction="mean")
        test_loss_U += ce_loss
        # Get labels
        y_hat_labels = y_hat.softmax(dim=1).max(dim=1)[1]
        acc = accuracy_score(y.cpu().numpy(), y_hat_labels.cpu().numpy(), normalize=True)
        test_accuracy_U += acc
 
test_loss_U = test_loss_U / len(test_loader)
test_loss_U = test_loss_U.cpu().numpy()
test_accuracy_U = test_accuracy_U/len(test_loader)

print(f"Test loss for IID based on U embedding: {test_loss_U}")
print(f"Test accuracy for IID based on U embedding: {test_accuracy_U}")



# """
# ### Classify IID based on cell-state + U latent
# """

# # %%
# Classifier = create_mlp(
#     input_size=myLIVI.hparams.z_dim+myLIVI.hparams.n_gxc_factors,
#     output_size=myLIVI.hparams.y_dim,
#     hidden_dims=[1000, 5000, 2000],
#     layer_norm=False,
#     device="cuda:0",
# )
# Classifier

# # %%
# device="cuda:0"
# batch_size=1024
# train_epochs=50
# optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-4, weight_decay=0.0)

# # %%
# generator = torch.Generator().manual_seed(32)

# X = torch.cat([zbase, U], dim=1)

# dataset = TensorDataset(X, Y)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator)
# train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# torch.save(test_loader, "/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/test_loader_z-U.pt")


# # %%
# train_error = {}
# validation_error = {}
# best_epoch = 1
# best_loss = 10e10

# for epoch in range(1, train_epochs+1):
#     batch_ce = 0.0

#     for x, y in train_loader:
#         # Pass through VAE
#         x = x.to(device, dtype=torch.float)
#         y = y.to(device, dtype=torch.long)

#         ce_loss = F.cross_entropy(Classifier(x), y)
#         batch_ce += ce_loss.item()

#         # Backprop and optimize
#         optimizer.zero_grad()
#         try:
#             ce_loss.backward()
#         #   nn.utils.clip_grad_norm_(model.parameters(), 10000)
#         except RuntimeError:
#             print("Epoch: {}   batch: {}".format(epoch, b))

#         optimizer.step()

#     # Mean epoch loss over batches
#     epoch_loss = batch_ce / len(train_loader)
#     train_error[f"Epoch_{epoch}"] = epoch_loss

#     # Calculate validation loss for epoch
#     val_loss = 0.0
#     with torch.no_grad():
#         Classifier.eval()
#         for x, y in val_loader:
#             x = x.to(device, dtype=torch.float)
#             y = y.to(device, dtype=torch.long)
#             ce_loss_val = F.cross_entropy(Classifier(x), y)
#             val_loss += ce_loss_val.item()

#     # Mean validation loss for the epoch over batches
#     val_loss = val_loss / len(val_loader)
#     if epoch % 5 == 0:
#         print(
#             f"Epoch {epoch}   Current Train Loss: {epoch_loss:.5f}, Current Val Loss: {val_loss:.5f}, Current best epoch: {best_epoch}"
#         )
#     validation_error[f"Epoch_{epoch}"] = val_loss

#     if val_loss <= best_loss:
#         best_loss = val_loss
#         best_epoch = epoch

#         torch.save(Classifier, "/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state-and-U_best-epoch.pt")

#     torch.save(Classifier, "/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state-and-U.pt")


# # %%
# Classifier = torch.load("/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state-and-U_best-epoch.pt")


# # %%
# test_loss_z_U = 0.0
# test_accuracy_z_U = 0.0

# with torch.no_grad():
#     Classifier.eval()
#     for x, y in test_loader:
#         x = x.to(device, dtype=torch.float)
#         y = y.to(device, dtype=torch.long)
#         y_hat = Classifier(x)
#         ce_loss = F.cross_entropy(y_hat, y, reduction="mean")
#         test_loss_z_U += ce_loss
#         # Get labels
#         y_hat_labels = y_hat.softmax(dim=1).max(dim=1)[1]
#         acc = accuracy_score(y.cpu().numpy(), y_hat_labels.cpu().numpy(), normalize=True)
#         test_accuracy_z_U += acc
 
# test_loss_z_U = test_loss_z_U / len(test_loader)
# test_loss_z_U = test_loss_z_U.cpu().numpy()
# test_accuracy_z_U = test_accuracy_z_U/len(test_loader)

# print(f"Test loss for IID based on cell-state and U: {test_loss_z_U}")
# print(f"Test accuracy for IID based on cell-state and U: {test_accuracy_z_U}")



"""
### Classify IID based on V embedding
"""

# %%
Classifier = create_mlp(
    input_size=myLIVI.hparams.n_persistent_factors,
    output_size=myLIVI.hparams.y_dim,
    hidden_dims=[100, 1000, 5000],
    layer_norm=False,
    device="cuda:0",
)
Classifier

# %%
device="cuda:0"
batch_size=1024
train_epochs=50
optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-4, weight_decay=0.0)

# %%
generator = torch.Generator().manual_seed(32)

dataset = TensorDataset(V, Y)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
torch.save(test_loader, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/test_loader_V.pt")


# %%
train_error = {}
validation_error = {}
best_epoch = 1
best_loss = 10e10

for epoch in range(1, train_epochs+1):
    batch_ce = 0.0

    for x, y in train_loader:
        # Pass through VAE
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

        ce_loss = F.cross_entropy(Classifier(x), y)
        batch_ce += ce_loss.item()

        # Backprop and optimize
        optimizer.zero_grad()
        try:
            ce_loss.backward()
        #   nn.utils.clip_grad_norm_(model.parameters(), 10000)
        except RuntimeError:
            print("Epoch: {}   batch: {}".format(epoch, b))

        optimizer.step()

    # Mean epoch loss over batches
    epoch_loss = batch_ce / len(train_loader)
    train_error[f"Epoch_{epoch}"] = epoch_loss

    # Calculate validation loss for epoch
    val_loss = 0.0
    with torch.no_grad():
        Classifier.eval()
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            ce_loss_val = F.cross_entropy(Classifier(x), y)
            val_loss += ce_loss_val.item()

    # Mean validation loss for the epoch over batches
    val_loss = val_loss / len(val_loader)
    if epoch % 5 == 0:
        print(
            f"Epoch {epoch}   Current Train Loss: {epoch_loss:.5f}, Current Val Loss: {val_loss:.5f}, Current best epoch: {best_epoch}"
        )
    validation_error[f"Epoch_{epoch}"] = val_loss

    if val_loss <= best_loss:
        best_loss = val_loss
        best_epoch = epoch

        torch.save(Classifier, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-V-embedding_best-epoch.pt")

    torch.save(Classifier, "/omics/groups/OE0540/internal/projects/LIVI/OneK1K/lassifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-V-embedding.pt")


# %%
Classifier = torch.load("/omics/groups/OE0540/internal/projects/LIVI/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-V-embedding_best-epoch.pt")


# %%
test_loss_V = 0.0
test_accuracy_V = 0.0

with torch.no_grad():
    Classifier.eval()
    for x, y in test_loader:
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        y_hat = Classifier(x)
        ce_loss = F.cross_entropy(y_hat, y, reduction="mean")
        test_loss_V += ce_loss
        # Get labels
        y_hat_labels = y_hat.softmax(dim=1).max(dim=1)[1]
        acc = accuracy_score(y.cpu().numpy(), y_hat_labels.cpu().numpy(), normalize=True)
        test_accuracy_V += acc
 
test_loss_V = test_loss_V / len(test_loader)
test_loss_V = test_loss_V.cpu().numpy()
test_accuracy_V = test_accuracy_V/len(test_loader)

print(f"Test loss for IID based on V embedding: {test_loss_V}")
print(f"Test accuracy for IID based on V embedding: {test_accuracy_V}")


# # %%
# """
# ### Classify IID based on cell-state + U + V latent
# """


# # %%
# Classifier = create_mlp(
#     input_size=myLIVI.hparams.z_dim+myLIVI.hparams.n_gxc_factors+myLIVI.hparams.n_persistent_factors,
#     output_size=myLIVI.hparams.y_dim,
#     hidden_dims=[100, 1000, 5000],
#     layer_norm=False,
#     device="cuda:0",
# )
# Classifier

# # %%
# device="cuda:0"
# batch_size=1024
# train_epochs=50
# optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-4, weight_decay=0.0)

# # %%
# generator = torch.Generator().manual_seed(32)

# X = torch.cat([zbase, U, V], dim=1)

# dataset = TensorDataset(X, Y)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator)
# train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.95, 0.05], generator)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# torch.save(test_loader, "/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/test_loader_z-U-V.pt")


# # %%
# train_error = {}
# validation_error = {}
# best_epoch = 1
# best_loss = 10e10

# for epoch in range(1, train_epochs+1):
#     batch_ce = 0.0

#     for x, y in train_loader:
#         # Pass through VAE
#         x = x.to(device, dtype=torch.float)
#         y = y.to(device, dtype=torch.long)

#         ce_loss = F.cross_entropy(Classifier(x), y)
#         batch_ce += ce_loss.item()

#         # Backprop and optimize
#         optimizer.zero_grad()
#         try:
#             ce_loss.backward()
#         #   nn.utils.clip_grad_norm_(model.parameters(), 10000)
#         except RuntimeError:
#             print("Epoch: {}   batch: {}".format(epoch, b))

#         optimizer.step()

#     # Mean epoch loss over batches
#     epoch_loss = batch_ce / len(train_loader)
#     train_error[f"Epoch_{epoch}"] = epoch_loss

#     # Calculate validation loss for epoch
#     val_loss = 0.0
#     with torch.no_grad():
#         Classifier.eval()
#         for x, y in val_loader:
#             x = x.to(device, dtype=torch.float)
#             y = y.to(device, dtype=torch.long)
#             ce_loss_val = F.cross_entropy(Classifier(x), y)
#             val_loss += ce_loss_val.item()

#     # Mean validation loss for the epoch over batches
#     val_loss = val_loss / len(val_loader)
#     if epoch % 5 == 0:
#         print(
#             f"Epoch {epoch}   Current Train Loss: {epoch_loss:.5f}, Current Val Loss: {val_loss:.5f}, Current best epoch: {best_epoch}"
#         )
#     validation_error[f"Epoch_{epoch}"] = val_loss

#     if val_loss <= best_loss:
#         best_loss = val_loss
#         best_epoch = epoch

#         torch.save(Classifier, "/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state-and-U-and-V_best-epoch.pt")

#     torch.save(Classifier, "/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state-and-U-and-V.pt")


# # %%
# Classifier = torch.load("/omics/groups/OE0540/internal_temp/users/danai/pc07/data_pc07/Data/OneK1K/Classifier_to_predict_IID_based_on_latent_space_10K-HVG-HEX_Fig2/classifier_IID-based-on-cell-state-and-U-and-V_best-epoch.pt")


# # %%
# test_loss_z_U_V = 0.0
# test_accuracy_z_U_V = 0.0

# with torch.no_grad():
#     Classifier.eval()
#     for x, y in test_loader:
#         x = x.to(device, dtype=torch.float)
#         y = y.to(device, dtype=torch.long)
#         y_hat = Classifier(x)
#         ce_loss = F.cross_entropy(y_hat, y, reduction="mean")
#         test_loss_z_U_V += ce_loss
#         # Get labels
#         y_hat_labels = y_hat.softmax(dim=1).max(dim=1)[1]
#         acc = accuracy_score(y.cpu().numpy(), y_hat_labels.cpu().numpy(), normalize=True)
#         test_accuracy_z_U_V += acc
 
# test_loss_z_U_V = test_loss_z_U_V / len(test_loader)
# test_loss_z_U_V = test_loss_z_U_V.cpu().numpy()
# test_accuracy_z_U_V = test_accuracy_z_U_V/len(test_loader)

# print(f"Test loss for IID based on cell-state and U and V: {test_loss_z_U_V}")
# print(f"Test accuracy for IID based on cell-state and U and V: {test_accuracy_z_U_V}")


# %%
"""
### Plot
"""

# results = pd.DataFrame.from_dict({"input": ["cell-state latent", "U latent", "cell-state + U latent", "V latent", "cell-state + U + V latent"], 
#                                   "test_accuracy": [test_accuracy_cell_state, test_accuracy_U, test_accuracy_z_U, test_accuracy_V, test_accuracy_z_U_V]}, 
#                                  orient="columns")

results = pd.DataFrame.from_dict({"input": ["cell-state latent", "U latent", "V latent"], 
                                  "test_accuracy": [test_accuracy_cell_state, test_accuracy_U, test_accuracy_V]}, 
                                 orient="columns")
results.to_csv(os.path.join(model_results_dir, "results_donor_classification_based_on_latent.tsv"),
               sep="\t", index=True, header=True)


sns.barplot(results, x="test_accuracy", y="input", color="salmon", rasterized=True)
plt.xlabel("Test set accuracy", fontsize=14)
plt.ylabel("Input latent embedding", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Donor classification based on LIVI latent space\n", fontsize=14)
plt.savefig(os.path.join(model_results_dir, "Figures", "Donor classification based on latent space_z_U_V.png".replace(" ", "-")), 
            dpi=400, bbox_inches="tight", transparent=True)
plt.savefig(os.path.join(model_results_dir, "Figures", "Donor classification based on latent space_z_U_V.svg".replace(" ", "-")), 
            dpi=400, bbox_inches="tight", transparent=True)
