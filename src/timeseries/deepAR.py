from pytorch_forecasting import DeepAR

class deepAR:
    
    def __new__(cls,dataset,**kwargs):
        
        return DeepAR.from_dataset(dataset,**kwargs)
    
    @classmethod
    
    def load_from_checkpoint(cls, path):
        
        return DeepAR.load_from_checkpoint(path)