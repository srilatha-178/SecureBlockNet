from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def save_confusion_matrix(y_true,y_pred,class_names,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,7))
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred,display_labels=class_names,xticks_rotation=45,ax=ax,colorbar=False)
    fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

def save_training_curves(history,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot(history.history.get('loss',[]),label='Train Loss'); ax.plot(history.history.get('val_loss',[]),label='Validation Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

def save_attention_heatmap(weights,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    w=np.asarray(weights)
    fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(w,aspect='auto'); ax.set_xlabel('Temporal Step'); ax.set_ylabel('Sample'); fig.colorbar(im,ax=ax,label='Attention Weight'); fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)
