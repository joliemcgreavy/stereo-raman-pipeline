"""
Auto-download datasets for the Streamlit dashboard.

Called once at app startup. Checks whether each dataset is present
and downloads it if not. Uses only standard-library urllib so no extra
dependencies are needed beyond what's already in requirements.txt.

Both datasets are open-access (CC BY 4.0) and hosted on Figshare — the
same infrastructure that hosts the raw files we already use locally.
Download only triggers on Streamlit Community Cloud (or any environment
where the files are absent); local runs with data already present skip
all downloads instantly.
"""

from __future__ import annotations

import io
import zipfile
import urllib.request
from pathlib import Path

# Project root → data directory
DATA_ROOT = Path(__file__).parent

# ── Download URLs ──────────────────────────────────────────────────────────
RAMAN_FILES = {
    'raw_COVID.txt':    'https://ndownloader.figshare.com/files/22386432',
    'raw_Healthy.txt':  'https://ndownloader.figshare.com/files/22386435',
    'data.mat':         'https://ndownloader.figshare.com/files/22386411',
}
SERV_CT_URL = 'https://ndownloader.figshare.com/files/47857471'


def _fetch(url: str, desc: str = '') -> bytes:
    """Download a URL and return its bytes."""
    req = urllib.request.Request(url, headers={'User-Agent': 'stereo-raman-pipeline/1.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def ensure_raman_data(status=None) -> bool:
    """
    Download the Yin et al. (2021) COVID-19 serum Raman dataset if absent.

    Files go to data/raw/raman_covid/. The Raman shift axis is extracted
    from data.mat using scipy (already in requirements.txt).

    Returns True if data is available after the call.
    """
    raman_dir = DATA_ROOT / 'raw' / 'raman_covid'
    needed = ['raw_COVID.txt', 'raw_Healthy.txt', 'Raman_shift.txt']

    if all((raman_dir / f).exists() for f in needed):
        return True

    raman_dir.mkdir(parents=True, exist_ok=True)

    try:
        for filename, url in RAMAN_FILES.items():
            dest = raman_dir / filename
            if dest.exists():
                continue
            if status:
                status.update(label=f'Downloading {filename}...')
            data = _fetch(url)
            dest.write_bytes(data)

        # Extract Raman shift axis from data.mat
        shift_path = raman_dir / 'Raman_shift.txt'
        if not shift_path.exists():
            import scipy.io
            import numpy as np
            mat = scipy.io.loadmat(str(raman_dir / 'data.mat'))
            np.savetxt(str(shift_path), mat['wave_number'].ravel())

        return True

    except Exception as e:
        if status:
            status.update(label=f'Raman download failed: {e}')
        return False


def ensure_serv_ct(status=None) -> bool:
    """
    Download and extract the SERV-CT stereo dataset if absent.

    Extracts to data/raw/serv_ct/SERV-CT/. The zip is ~35MB and contains
    16 stereo image pairs with calibration and CT ground-truth depth maps.

    Returns True if data is available after the call.
    """
    serv_dir = DATA_ROOT / 'raw' / 'serv_ct' / 'SERV-CT'

    if serv_dir.exists() and any(serv_dir.iterdir()):
        return True

    serv_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        if status:
            status.update(label='Downloading SERV-CT (~35MB)...')
        raw = _fetch(SERV_CT_URL)

        if status:
            status.update(label='Extracting SERV-CT...')
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(str(serv_dir.parent))

        return True

    except Exception as e:
        if status:
            status.update(label=f'SERV-CT download failed: {e}')
        return False


def ensure_all(status=None) -> dict[str, bool]:
    """Download all datasets. Returns availability dict."""
    raman_ok   = ensure_raman_data(status)
    serv_ct_ok = ensure_serv_ct(status)
    if status:
        status.update(label='Datasets ready.', state='complete')
    return {'raman': raman_ok, 'serv_ct': serv_ct_ok}
