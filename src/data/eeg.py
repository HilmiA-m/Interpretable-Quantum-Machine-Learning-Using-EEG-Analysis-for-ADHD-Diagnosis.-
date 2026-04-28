import glob
import numpy as np
import mne
from src import config


def _infer_fs(df):
    for col in ("timestamps", "Timestamp"):
        if col in df.columns:
            ts = df[col].values.astype(float)
            d  = np.diff(ts)
            d  = d[(d > 0) & (d < 1.0)]
            if len(d):
                fs = int(round(1.0 / np.median(d)))
                if 64 <= fs <= 2000:
                    return fs
    return 500


def _hjorth(sig):
    var0 = np.var(sig) + 1e-12
    d1   = np.diff(sig); var1 = np.var(d1) + 1e-12
    d2   = np.diff(d1);  var2 = np.var(d2) + 1e-12
    mob  = np.sqrt(var1 / var0)
    return mob, np.sqrt(var2 / var1) / (mob + 1e-12)


def extract_eeg_features(csv_path):
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        SKIP    = {"timestamps", "Counter", "Interpolated", "Battery", "Marker",
                   "Packet", "TRIGGER", "Packet Counter"}
        SKIP_KW = ("CQ.", "EQ.", "MOT.", "PM.", "POW.", "ACC", "ExG", "A1", "A2")
        eeg_cols = [
            c for c in df.columns
            if c not in SKIP
            and not any(k in c for k in SKIP_KW)
            and df[c].dtype in (float, "float64", "int64", "float32")
        ]
        if not eeg_cols:
            return None
        data  = df[eeg_cols].values.T.astype(float)
        valid = np.array([data[i].std() > 1e-6 for i in range(data.shape[0])])
        if valid.sum() == 0:
            return None
        data_v = data[valid]
        fs     = _infer_fs(df)
        info   = mne.create_info(data_v.shape[0], fs, ch_types="eeg")
        raw    = mne.io.RawArray(data_v, info, verbose=False)
        raw.filter(1, 45, fir_design="firwin", verbose=False)
        n_fft = max(64, min(fs * 4, data_v.shape[1] // 2))
        psd_o = raw.compute_psd(method="welch", fmin=1, fmax=45,
                                n_fft=n_fft, verbose=False)
        freqs = psd_o.freqs
        P     = psd_o.get_data()
        bp    = {b: P[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
                 for b, (lo, hi) in config.BANDS.items()}

        def pad29(a):
            a = (a[:config.N_CGX] if len(a) >= config.N_CGX
                 else np.pad(a, (0, config.N_CGX - len(a)), constant_values=np.nan))
            return a.astype(float)

        def safe_r(a29, idx):
            v = a29[idx]
            return float(np.nanmean(v)) if not np.all(np.isnan(v)) else 0.0

        tbr_ch   = pad29(bp["theta"] / (bp["beta"] + 1e-12))
        tbr_v    = tbr_ch[~np.isnan(tbr_ch)]
        tbr_mean = float(tbr_v.mean()) if len(tbr_v) else 0.0
        tbr_std  = float(tbr_v.std())  if len(tbr_v) else 0.0

        a29 = pad29(bp["alpha"]); t29 = pad29(bp["theta"])
        lfa = safe_r(a29, config.LEFT_FRONTAL);  rfa = safe_r(a29, config.RIGHT_FRONTAL)
        lft = safe_r(t29, config.LEFT_FRONTAL);  rft = safe_r(t29, config.RIGHT_FRONTAL)
        alpha_asym = float(np.log(rfa / (lfa + 1e-12) + 1e-12))
        theta_asym = float(np.log(rft / (lft + 1e-12) + 1e-12))
        occ_alpha  = safe_r(a29, config.OCCIPITAL)

        def sp_ent(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            p    = P[:, mask].mean(axis=0)
            p    = p / (p.sum() + 1e-12)
            return float(-np.sum(p * np.log(p + 1e-12)))

        ra = raw.get_data()
        fi = [i for i in config.LEFT_FRONTAL + config.RIGHT_FRONTAL if i < ra.shape[0]]
        mobs, comps = [], []
        for i in fi:
            m, c = _hjorth(ra[i])
            mobs.append(m); comps.append(c)
        hm  = float(np.mean(mobs))  if mobs  else 0.0
        hc  = float(np.mean(comps)) if comps else 0.0
        tar = float((bp["theta"] / (bp["alpha"] + 1e-12)).mean())

        feats = np.array([
            tbr_mean, tbr_std,
            alpha_asym, theta_asym,
            occ_alpha,
            sp_ent(4, 8), sp_ent(13, 30),
            hm, hc, tar,
            bp["delta"].mean(), bp["theta"].mean(),
            bp["alpha"].mean(), bp["beta"].mean(),
            bp["gamma"].mean(),
        ], dtype=float)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def find_eeg_file(uid, lvl):
    for pat in [
        f"{config.DATA_ROOT}/{uid}/{lvl}/*/{uid}_EEG_CGX_*.csv",
        f"{config.DATA_ROOT}/{uid}/{lvl}*/*/{uid}_EEG_CGX_*.csv",
    ]:
        files = glob.glob(pat)
        if files:
            return files[0]
    return None
