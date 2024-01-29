#finding pca

import rasterio
import numpy as np
import matplotlib.pyplot as plt


bands = [#"T43SBT_20231116T055109_AOT_20m.jp2",
"T43SBT_20231116T055109_B01_20m.jp2",
"T43SBT_20231116T055109_B02_20m.jp2",
"T43SBT_20231116T055109_B03_20m.jp2",
"T43SBT_20231116T055109_B04_20m.jp2",
"T43SBT_20231116T055109_B05_20m.jp2",
"T43SBT_20231116T055109_B06_20m.jp2",
"T43SBT_20231116T055109_B07_20m.jp2",
"T43SBT_20231116T055109_B8A_20m.jp2",
"T43SBT_20231116T055109_B11_20m.jp2",
"T43SBT_20231116T055109_B12_20m.jp2",
"T43SBT_20231116T055109_SCL_20m.jp2",
"T43SBT_20231116T055109_TCI_20m.jp2",
"T43SBT_20231116T055109_WVP_20m.jp2"
]

# Load and stack data
data = []
for band in bands:
    with rasterio.open(f"D:/GIKI/5th semester/ES-304 -Linear Algebra 2/CEP/S2B_MSIL2A_20231116T055109_N0509_R048_T43SBT_20231116T082205.SAFE/GRANULE/L2A_T43SBT_A034966_20231116T055536/IMG_DATA/R20m/{band}") as src:


        data.append(src.read(1))

data = np.stack(data, axis=-1)
# Standardize data
mean = np.mean(data, axis=(0, 1))

centered_data = data - mean
standardized_data = centered_data / np.std(centered_data, axis=(0, 1))

# Calculate covariance matrix
covariance_matrix = np.cov(standardized_data.reshape(-1, standardized_data.shape[-1]).T)
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

#Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print("eigenvectors",eigenvectors)
print("type(eigenvectors)",type(eigenvectors))


# Select desired number of principal components
n_components = 2

# Select top n_components eigenvectors
principal_components = eigenvectors[:, :n_components]

# Project data onto principal components
projected_data = standardized_data @ principal_components
# If needed, inverse transform to the original space


# Calculate explained variance ratio
explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

# Print results
print("Original data shape:", data.shape)
print("Standardized data shape:", standardized_data.shape)
print("Eigenvalues:", eigenvalues)
print("Explained variance ratio:", explained_variance_ratio)
print("Projected data shape:", projected_data.shape)


# Visualize standardized data for selected bands
plt.imshow(standardized_data[:, :, [2, 1, 0]])
plt.title("Standardized RGB Image")
plt.show()

#Image after pca
inverse_transformed_data = projected_data @ principal_components.T
plt.imshow(inverse_transformed_data[:, :, [2, 1, 0]])
plt.title('Image After PCA')
plt.show()

# Visualize scatter plot of the first two principal components
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.title("Projection onto First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# error analysis

# Loop over different numbers of principal components
for n_components in range(1, min(data.shape[-1], 6)):  # Example: consider up to 5 components
    # Select top n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]
    print("principal_components",principal_components)
    # Project data onto principal components
    projected_data = standardized_data @ principal_components
    print("projected_data",projected_data)
    # Reconstruct data from projected data
    reconstructed_data = projected_data @ principal_components.T
    print("reconstructed_data",reconstructed_data)
    # Calculate reconstruction error (MSE as an example)
    mse = np.mean(np.square(standardized_data - reconstructed_data))

    # Print or store information loss metrics
    print(f"Number of Components: {n_components}, MSE: {mse}")

