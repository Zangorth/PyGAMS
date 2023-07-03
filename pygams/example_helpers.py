####################
# Feature Pipeline #
####################
class FeaturePipeline():
    def __init__(self, features, scaler, imputer):
        self.features = features
        
        return None
    
    def fit(self, df):
        return None
    
    def transform(self, df):
        return df[self.features]