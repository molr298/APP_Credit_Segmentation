from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer



def normalizes(data):
  mm = MinMaxScaler()
  df_transform = pd.DataFrame(mm.fit_transform(data))
  pt = PowerTransformer()
  df_transform = pd.DataFrame(pt.fit_transform(df_transform))

  return df_transform


def PCA(data):
  # Transform into Array
  df_transform_PCA = np.asarray(data.fillna(data.mean()))
  
  # Applying PCA
  pca = PCA(n_components=2, random_state=24)
  df_transform_PCA = pca.fit_transform(df_transform_PCA)

  return df_transform_PCA
