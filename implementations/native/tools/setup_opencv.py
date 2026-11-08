"""Download the pinned official OpenCV Windows SDK; extract without executing it."""
import argparse,hashlib,json,shutil,subprocess,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='5.0.0');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];deps=root/'third_party';deps.mkdir(exist_ok=True)
    url=f'https://api.github.com/repos/opencv/opencv/releases/tags/{a.version}'
    req=urllib.request.Request(url,headers={'User-Agent':'flir-agc-native-reproduction'})
    release=json.load(urllib.request.urlopen(req))
    matches=[v for v in release['assets'] if v['name']==f'opencv-{a.version}-windows.exe']
    if len(matches)!=1:raise RuntimeError('Official Windows asset not uniquely found: '+str([v['name'] for v in release['assets']]))
    asset=matches[0];archive=deps/asset['name']
    if not archive.exists():
        print('Downloading',asset['browser_download_url'],flush=True)
        urllib.request.urlretrieve(asset['browser_download_url'],archive)
    with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    expected=asset.get('digest')
    if expected and expected!='sha256:'+actual:raise RuntimeError('SDK SHA-256 mismatch')
    record={'version':a.version,'release_url':release['html_url'],'asset_url':asset['browser_download_url'],
            'size_bytes':archive.stat().st_size,'sha256':actual,'github_digest':expected}
    (deps/'opencv_download.json').write_text(json.dumps(record,indent=2))
    seven=shutil.which('7z') or r'C:\Program Files\7-Zip\7z.exe'
    if not Path(seven).is_file():raise RuntimeError('7-Zip is required to extract the official self-extracting archive without running it')
    if not (deps/'opencv'/'build'/'OpenCVConfig.cmake').exists():
        subprocess.run([seven,'x',str(archive),'-o'+str(deps),'-y'],check=True,stdout=subprocess.DEVNULL)
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
