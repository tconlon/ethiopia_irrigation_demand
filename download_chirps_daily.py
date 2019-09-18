import sys
import ftplib
import os
from ftplib import FTP
from main import get_args
import zipfile
import glob
import gzip
import shutil


def fetchFiles(ftp, path, destination, overwrite=True):
    '''Fetch a whole folder from ftp. \n
    Parameters
    ----------
    ftp         : ftplib.FTP object
    path        : string ('/dir/folder/')
    destination : string ('D:/dir/folder/') folder where the files will be saved
    overwrite   : bool - Overwrite file if already exists.
    '''
    try:
        ftp.cwd(path)
        os.mkdir(destination + path)
        print('New folder made: ' + destination + path)
    except OSError:
        # folder already exists at the destination
        pass
    except ftplib.error_perm:
        # invalid entry (ensure input form: "/dir/folder/")
        print("error: could not change to " + path)
        sys.exit("ending session")

    # Create list of images in CHIRPS FTP folder
    filelist = [i for i in ftp.nlst()]

    for file in filelist:

        fullpath = os.path.join(destination, file)
        if (not overwrite and os.path.isfile(fullpath)):
            continue
        else:
            with open(fullpath, 'wb') as f:
                # Download zipped file
                ftp.retrbinary('RETR ' + file, f.write)
            print(file + ' downloaded')

def unzip(dir):
    # Unzip the downloaded imagery and remove zipped file
    for item in glob.glob(dir + '/*.gz'):  # loop through items in dir
        with gzip.open(item, 'rb') as f_in:
            with open(item[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(item)


if __name__ == "__main__":
    args = get_args()

    # Year of desired CHIRPS data
    year = '2018'

    ftp = FTP('ftp.chg.ucsb.edu')
    ftp.login()

    # The source of CHIRPS data
    source = '/pub/org/chg/products/CHIRPS-2.0/africa_daily/tifs/p05/' + year +'/'

    # Create destination folder for downloaded image
    dest = os.path.join(args.chirps_dir, year)
    if not os.path.exists(dest):
        os.mkdir(dest)

    # Uncomment these two following lines to download more imagery
    # fetchFiles(ftp, source, dest, overwrite=True)
    # ftp.quit()

    # This line unzips all the downloaded imagery
    unzip(dest)