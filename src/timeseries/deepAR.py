from pytorch_forecasting import DeepAR

class DeepAR:
    
    def __new__(cls,dataset,**kwargs):
        
        return DeepAR.from_dataset(dataset,**kwargs)