import gdown

url = (
    "https://drive.google.com/file/d/1Ljv_s7HrLR2nAK4kvA87EtYRIqK2MChT/view?usp=sharing"
)
output_file = "vivos.tar.gz"
gdown.download(url, output_file, quiet=False, fuzzy=True)
