"""
analyze_age_bias.py
Computes the average signed difference (predicted - actual) per age bracket and per individual year.
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from models import AgeModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class SimpleAgeDataset(Dataset):
    def __init__(self, df, transform):
        self.fps = df['filepath'].values
        self.ages = df['age'].values.astype(np.float32)
        self.tf = transform
    def __len__(self): return len(self.fps)
    def __getitem__(self, i):
        try: img = Image.open(self.fps[i]).convert('RGB')
        except: img = Image.new('RGB', (320, 320), (0, 0, 0))
        return self.tf(img), torch.tensor(self.ages[i])

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf_o = T.Compose([T.Resize((320,320)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tf_f = T.Compose([T.Resize((320,320)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # Load both champion models
    mA = AgeModel(backbone_name='tf_efficientnetv2_s', head_type='dex', pretrained=False).to(device)
    mA.load_state_dict(torch.load('outputs/exp25_effnetv2s_dex_expected_age/best_model.pt', map_location=device)['model_state_dict'])
    mA.eval()

    mB = AgeModel(backbone_name='tf_efficientnetv2_s', head_type='hybrid', pretrained=False).to(device)
    mB.load_state_dict(torch.load('outputs/exp23_effnetv2s_utkface_supplement/best_model.pt', map_location=device)['model_state_dict'])
    mB.eval()

    df = pd.read_csv('manifest_p2_320_plus_utkface.csv')
    val = df[df['split'] == 'val'].reset_index(drop=True)

    ds_o = SimpleAgeDataset(val, tf_o)
    ds_f = SimpleAgeDataset(val, tf_f)
    lo = DataLoader(ds_o, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    lf = DataLoader(ds_f, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    preds, targets = [], []
    with torch.no_grad():
        for (io, ys), (if_, _) in tqdm(zip(lo, lf), total=len(lo), desc="Ensemble Inference"):
            io, if_ = io.to(device), if_.to(device)
            pA = 0.5 * mA(io)['pred_age'].cpu().numpy() + 0.5 * mA(if_)['pred_age'].cpu().numpy()
            pB = 0.5 * mB(io)['pred_age'].cpu().numpy() + 0.5 * mB(if_)['pred_age'].cpu().numpy()
            p = 0.5 * pA + 0.5 * pB
            preds.extend(p.tolist())
            targets.extend(ys.numpy().tolist())

    y_true = np.array(targets)
    y_pred = np.array(preds)
    signed_diff = y_pred - y_true
    abs_error = np.abs(signed_diff)

    # Bracket summary
    bins = [0, 12, 19, 35, 45, 60, 75, 100]
    labels = ['1-12', '13-19', '20-35', '36-45', '46-60', '61-75', '76-100']
    res = pd.DataFrame({'actual': y_true, 'predicted': y_pred, 'signed_diff': signed_diff, 'abs_error': abs_error})
    res['bracket'] = pd.cut(res['actual'], bins=bins, labels=labels)

    summary = res.groupby('bracket', observed=False).agg(
        Count=('abs_error', 'count'),
        Avg_Actual=('actual', 'mean'),
        Avg_Predicted=('predicted', 'mean'),
        Avg_Signed_Diff=('signed_diff', 'mean'),
        MAE=('abs_error', 'mean'),
        Median_Error=('abs_error', 'median')
    ).round(3)

    print('\n' + '=' * 95)
    print(' AVERAGE DIFFERENCE (Predicted - Actual) PER AGE BRACKET')
    print(' Positive = Over-predicts (model says older) | Negative = Under-predicts (model says younger)')
    print('=' * 95)
    print(summary.to_string())

    # Per-year for ages 1-20
    print('\n' + '=' * 85)
    print(' PER-YEAR BIAS ANALYSIS (Ages 1 to 20)')
    print('=' * 85)
    for a in range(1, 21):
        mask = (y_true >= a - 0.5) & (y_true < a + 0.5)
        if mask.sum() > 0:
            avg_pred = np.mean(y_pred[mask])
            avg_diff = np.mean(signed_diff[mask])
            mae_a = np.mean(abs_error[mask])
            print(f'  Age {a:3d}: Count={mask.sum():5d} | Avg Predicted={avg_pred:6.2f} | Avg Diff={avg_diff:+6.2f} | MAE={mae_a:5.2f}')

    # Per-year for ages 40-100
    print('\n' + '=' * 85)
    print(' PER-YEAR BIAS ANALYSIS (Ages 40 to 100)')
    print('=' * 85)
    for a in range(40, 101):
        mask = (y_true >= a - 0.5) & (y_true < a + 0.5)
        if mask.sum() > 0:
            avg_pred = np.mean(y_pred[mask])
            avg_diff = np.mean(signed_diff[mask])
            mae_a = np.mean(abs_error[mask])
            print(f'  Age {a:3d}: Count={mask.sum():5d} | Avg Predicted={avg_pred:6.2f} | Avg Diff={avg_diff:+6.2f} | MAE={mae_a:5.2f}')
