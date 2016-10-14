import numpy as np
from scipy import sparse

def semi_supervised(model, X, y, Xt, thresh=0.9, predict_proba=None):
	if predict_proba is None: predict_proba = model.predict_proba
	while True:
		# Train on expanded training set
		model.fit(X, y)
		
		# Test on unsure test set
		yt = model.predict(Xt)
		yp = np.max(predict_proba(Xt), axis=1)
		
		# Get testing examples the model is sure of
		good_idx = np.where(yp >= thresh)[0]
		bad_idx = np.where(yp < thresh)[0]
		
		# Break if there aren't any good samples
		if not good_idx.size: break
		
		# Add confident samples to training set
		X = sparse.vstack((X, Xt[good_idx,:]))
		y = np.hstack((y, yt[good_idx]))
		
		# Remove confident samples from test set
		Xt = Xt[bad_idx,:]

	return model

