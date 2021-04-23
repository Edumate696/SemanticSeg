import kaggle
import pathlib
import shutil

# You need to have ~/.kaggle/kaggle.json in your device.

competition_name = 'tgs-salt-identification-challenge'
out_path = pathlib.Path('Dataset')


def download(train: bool = True) -> None:
    fn = 'train' if train else 'test'
    print(f'[INFO] Downloading {fn} data.')
    kaggle.api.competition_download_file(competition_name, fn + '.zip', path=out_path / '.temp_storage',
                                         force=True, quiet=True)
    shutil.rmtree(out_path / fn, ignore_errors=True)
    print(f'[INFO] Extracting {fn} data.')
    shutil.unpack_archive(str(out_path / '.temp_storage' / fn) + '.zip', out_path / fn)
    shutil.rmtree(out_path / '.temp_storage', ignore_errors=True)
    print()


if __name__ == '__main__':
    download(train=True)
    download(train=False)
    print('Done')
